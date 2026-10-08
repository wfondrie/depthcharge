"""Serve mass spectra to neural networks."""

from __future__ import annotations

import copy
import itertools
import logging
import uuid
import warnings
from collections.abc import Generator, Iterable
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from cloudpathlib import AnyPath
from lance.sampler import Sampler
from lance.torch.data import LanceDataset
from lance.torch.dist import (
    get_dist_world_size,
    get_global_rank,
    get_global_world_size,
)
from torch.utils.data import IterableDataset

from .. import utils
from ..tokenizers import PeptideTokenizer
from . import arrow
from .parsers import hash_peak_file

LOGGER = logging.getLogger(__name__)


class SpectrumDataset(LanceDataset):
    """Store and access a collection of mass spectra.

    Parse and/or add mass spectra to an index in the
    [lance data format](https://lance.org).
    This format enables fast random access to spectra for training.
    This file is then served as a PyTorch IterableDataset, allowing
    spectra to be accessed efficiently for training and inference.
    This is accomplished using the
    [Lance PyTorch integration](https://lance.org/integrations/pytorch).

    The `batch_size` parameter for this class independent of the `batch_size`
    of the PyTorch DataLoader. Generally, we only want the former parameter to
    greater than 1. Batches are divided among DataLoader workers and
    distributed training processes, so that each spectrum is loaded once per
    epoch. When using DataLoader workers, start them with
    `multiprocessing_context="spawn"` or `"forkserver"`, because Lance is not
    safe to use in forked processes.

    Peak files are identified using a fingerprint of their contents
    (see `depthcharge.data.hash_peak_file()`), which is stored in the
    `peak_file_hash` column. Peak files that have already been added
    to the dataset are skipped.

    If a lance dataset already exists at `path`, only the peak files that it
    does not already contain are parsed and added to it, unless `overwrite`
    is `True`. This makes it fast to re-create a dataset with the same peak
    files. Note that depthcharge does not verify that the existing dataset was
    created with the same `parse_kwargs`, such as preprocessing. To use an
    existing lance dataset without adding spectra, use the `from_lance()`
    method.

    Parameters
    ----------
    spectra : polars.DataFrame, PathLike, or list of PathLike
        Spectra to add to this collection. These may be a DataFrame parsed
        with `depthcharge.spectra_to_df()`, parquet files created with
        `depthcharge.spectra_to_parquet()`, or a peak file in the mzML,
        mzXML, MGF, or Bruker TDF format. Additional spectra can be added
        later using the `.add_spectra()` method.
    batch_size : int
        The batch size to use for loading mass spectra. Note that this is
        independent from the batch size for the PyTorch DataLoader.
    path : PathLike, optional.
        The name and path of the lance dataset. If the path does
        not contain the `.lance` then it will be added.
        If `None`, a file will be created in a temporary directory.
    overwrite : bool, optional
        Replace the lance dataset at `path` if it exists, rather than
        adding new peak files to it.
    parse_kwargs : dict, optional
        Keyword arguments passed `depthcharge.spectra_to_stream()` for
        peak files that are provided. This argument has no effect for
        DataFrame or parquet file inputs.
    pad_fields : str or iterable of str, optional
        Additional list columns to pad into a single tensor for each batch,
        in the same manner as the `mz_array` and `intensity_array` columns.
        Each value in these columns must be a list of numbers. Missing
        columns are ignored.
    shuffle : bool, optional
        Shuffle the spectra in a new order each epoch. Blocks of `batch_size`
        consecutive spectra are read in a random order, then the spectra are
        shuffled within a buffer of 16 blocks. Shuffling cannot be combined
        with the `filter`, `sampler`, `samples`, `shard_granularity`, or
        `with_row_id` options of the `LanceDataset`.
    seed : int, optional
        The random seed for shuffling. If `None`, PyTorch's initial seed
        (`torch.initial_seed()`) is used. In distributed training, every
        process must use the same seed.
    **kwargs : dict
        Keyword arguments to initialize a
        `[lance.torch.data.LanceDataset](https://github.com/lance-format/lance/blob/92aa361099f42a40e9aa9f9915d041fe1dd30671/python/python/lance/torch/data.py#L177)`.

    Attributes
    ----------
    peak_files : list of str
    peak_file_hashes : list of str
    path : Path
    n_spectra : int
    dataset : lance.LanceDataset

    """

    def __init__(
        self,
        spectra: pl.DataFrame | PathLike | Iterable[PathLike],
        batch_size: int,
        path: PathLike | None = None,
        parse_kwargs: dict | None = None,
        pad_fields: str | Iterable[str] | None = None,
        overwrite: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
        **kwargs: dict,
    ) -> None:
        """Initialize a SpectrumDataset."""
        self._pad_fields = _get_pad_fields(pad_fields)
        self._parse_kwargs = {} if parse_kwargs is None else parse_kwargs
        self._init_kwargs = copy.copy(self._parse_kwargs)
        self._init_kwargs["batch_size"] = 128
        self._init_kwargs["progress"] = False

        self._tmpdir = None
        if path is None:
            # Create a random temporary file:
            self._tmpdir = TemporaryDirectory()
            path = Path(self._tmpdir.name) / f"{uuid.uuid4()}.lance"

        self._path = AnyPath(path)
        if self._path.suffix != ".lance":
            self._path = self._path.with_suffix(".lance")

        # Now parse spectra.
        if spectra is not None:
            spectra = utils.listify(spectra)
            if self._path.exists() and not overwrite:
                existing = lance.dataset(str(self._path))
                _check_appendable(existing, spectra, self._parse_kwargs)
                spectra = _filter_duplicates(
                    spectra,
                    existing=_get_peak_file_hashes(existing),
                    warn=False,
                )
                if spectra:
                    _append(existing, spectra, self._parse_kwargs)
            else:
                spectra = _filter_duplicates(spectra)
                batch = next(_get_records(spectra, **self._init_kwargs))
                lance.write_dataset(
                    _get_records(spectra, **self._parse_kwargs),
                    str(self._path),
                    mode="overwrite" if self._path.exists() else "create",
                    schema=batch.schema,
                )

        elif not self._path.exists():
            raise ValueError("No spectra were provided")

        dataset = lance.dataset(str(self._path))
        if "to_tensor_fn" not in kwargs:
            kwargs["to_tensor_fn"] = self._to_tensor

        sampler = _get_sampler(shuffle, seed, kwargs)
        if sampler is not None:
            kwargs["sampler"] = sampler

        super().__init__(dataset, batch_size, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch, which determines the order of shuffled spectra.

        The epoch is incremented each time the dataset is iterated over.
        However, DataLoader workers start from the epoch of the dataset in
        the main process, unless they are persistent. Without persistent
        workers, a new order is still used each epoch, except during
        distributed training. In that case, call this method before each
        epoch, or use `persistent_workers=True` with the DataLoader.

        Parameters
        ----------
        epoch : int
            The epoch number.

        """
        if isinstance(self.sampler, _SpectrumSampler):
            self.sampler.set_epoch(epoch)

    def add_spectra(
        self,
        spectra: pl.DataFrame | PathLike | Iterable[PathLike],
    ) -> SpectrumDataset:
        """Add mass spectrometry data to the lance dataset.

        Peak files that have already been added to the dataset, as
        determined by their `peak_file_hash`, are skipped with a warning.
        Depthcharge does not verify whether spectra from DataFrame or parquet
        inputs already exist in the lance dataset.

        Parameters
        ----------
        spectra : polars.DataFrame, PathLike, or list of PathLike
            Spectra to add to this collection. These may be a DataFrame parsed
            with `depthcharge.spectra_to_df()`, parquet files created with
            `depthcharge.spectra_to_parquet()`, or a peak file in the mzML,
            mzXML, MGF, or Bruker TDF format.

        """
        spectra = _filter_duplicates(
            utils.listify(spectra),
            existing=self.peak_file_hashes,
        )
        if not spectra:
            warnings.warn("No new spectra were added to the dataset.")
            return self

        self.dataset = _append(self.dataset, spectra, self._parse_kwargs)
        return self

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Access a mass spectrum.

        Parameters
        ----------
        idx : int
            The index of the index of the mass spectrum to look up.

        Returns
        -------
        dict
            A dictionary representing a row of the dataset. Each
            key is a column and the value is the value for that
            row. List columns are automatically converted to
            PyTorch tensors if the nested data type is compatible.

        """
        return self._to_tensor(self.dataset.take(utils.listify(idx)))

    def __del__(self) -> None:
        """Cleanup the temporary directory."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    @property
    def n_spectra(self) -> int:
        """The number of spectra in the Lance dataset."""
        return self.dataset.count_rows()

    @property
    def peak_files(self) -> list[str]:
        """The files currently in the lance dataset."""
        return (
            self.dataset.to_table(columns=["peak_file"])
            .column(0)
            .unique()
            .to_pylist()
        )

    @property
    def peak_file_hashes(self) -> list[str]:
        """The fingerprints of the peak files in the lance dataset."""
        return _get_peak_file_hashes(self.dataset)

    @property
    def path(self) -> Path:
        """The path to the underlying lance dataset."""
        return self._path

    @classmethod
    def from_lance(
        cls,
        path: PathLike,
        batch_size: int,
        parse_kwargs: dict | None = None,
        pad_fields: str | Iterable[str] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        **kwargs: dict,
    ) -> SpectrumDataset:
        """Load a previously created lance dataset.

        Parameters
        ----------
        path : PathLike
            The path of the lance dataset.
        batch_size : int
            The batch size to use for loading mass spectra. Note that this is
            independent from the batch size for the PyTorch DataLoader.
        parse_kwargs : dict, optional
            Keyword arguments passed `depthcharge.spectra_to_stream()` for
            peak files that are provided.
        pad_fields : str or iterable of str, optional
            Additional list columns to pad into a single tensor for each batch,
            in the same manner as the `mz_array` and `intensity_array` columns.
            Each value in these columns must be a list of numbers. Missing
            columns are ignored.
        shuffle : bool, optional
            Shuffle the spectra in a new order each epoch. Blocks of
            `batch_size` consecutive spectra are read in a random order, then
            the spectra are shuffled within a buffer of 16 blocks. Shuffling
            cannot be combined with the `filter`, `sampler`, `samples`,
            `shard_granularity`, or `with_row_id` options of the
            `LanceDataset`.
        seed : int, optional
            The random seed for shuffling. If `None`, PyTorch's initial seed
            (`torch.initial_seed()`) is used. In distributed training, every
            process must use the same seed.
        **kwargs : dict
            Keyword arguments to initialize a
            `[lance.torch.data.LanceDataset](https://github.com/lance-format/lance/blob/92aa361099f42a40e9aa9f9915d041fe1dd30671/python/python/lance/torch/data.py#L177)`.

        Returns
        -------
        SpectrumDataset
            The dataset of mass spectra.

        """
        return cls(
            spectra=None,
            batch_size=batch_size,
            path=path,
            parse_kwargs=parse_kwargs,
            pad_fields=pad_fields,
            shuffle=shuffle,
            seed=seed,
            **kwargs,
        )

    def _to_tensor(
        self,
        batch: pa.RecordBatch,
        **ignored: dict[Any],
    ) -> dict[str, torch.Tensor | list[str | torch.Tensor]]:
        """Convert a record batch to tensors.

        Parameters
        ----------
        batch : pyarrow.RecordBatch
            The batch of data.
        **ignored : dict[Any]
            Ignored keyword arguments to maintain compatibility with
            pylance.

        Returns
        -------
        dict of str to tensors or lists
            The batch of data as a Python dict.

        """
        return _to_tensor(batch, self._pad_fields)


class AnnotatedSpectrumDataset(SpectrumDataset):
    """Store and access a collection of annotated mass spectra.

    Parse and/or add mass spectra to an index in the
    [lance data format](https://lance.org).
    This format enables fast random access to spectra for training.
    This file is then served as a PyTorch IterableDataset, allowing
    spectra to be accessed efficiently for training and inference.
    This is accomplished using the
    [Lance PyTorch integration](https://lance.org/integrations/pytorch).

    The `batch_size` parameter for this class independent of the `batch_size`
    of the PyTorch DataLoader. Generally, we only want the former parameter to
    greater than 1. Batches are divided among DataLoader workers and
    distributed training processes, so that each spectrum is loaded once per
    epoch. When using DataLoader workers, start them with
    `multiprocessing_context="spawn"` or `"forkserver"`, because Lance is not
    safe to use in forked processes.

    Peak files are identified using a fingerprint of their contents
    (see `depthcharge.data.hash_peak_file()`), which is stored in the
    `peak_file_hash` column. Peak files that have already been added
    to the dataset are skipped.

    If a lance dataset already exists at `path`, only the peak files that it
    does not already contain are parsed and added to it, unless `overwrite`
    is `True`. This makes it fast to re-create a dataset with the same peak
    files. Note that depthcharge does not verify that the existing dataset was
    created with the same `parse_kwargs`, such as preprocessing. To use an
    existing lance dataset without adding spectra, use the `from_lance()`
    method.

    Parameters
    ----------
    spectra : polars.DataFrame, PathLike, or list of PathLike
        Spectra to add to this collection. These may be a DataFrame parsed
        with `depthcharge.spectra_to_df()`, parquet files created with
        `depthcharge.spectra_to_parquet()`, or a peak file in the mzML,
        mzXML, MGF, or Bruker TDF format. Additional spectra can be added
        later using the `.add_spectra()` method.
    annotations : str
        The column name containing the annotations.
    tokenizer : PeptideTokenizer
        The tokenizer used to transform the annotations into PyTorch
        tensors.
    batch_size : int
        The batch size to use for loading mass spectra. Note that this is
        independent from the batch size for the PyTorch DataLoader.
    path : PathLike, optional.
        The name and path of the lance dataset. If the path does
        not contain the `.lance` then it will be added.
        If ``None``, a file will be created in a temporary directory.
    overwrite : bool, optional
        Replace the lance dataset at `path` if it exists, rather than
        adding new peak files to it.
    parse_kwargs : dict, optional
        Keyword arguments passed `depthcharge.spectra_to_stream()` for
        peak files that are provided. This argument has no effect for
        DataFrame or parquet file inputs.
    pad_fields : str or iterable of str, optional
        Additional list columns to pad into a single tensor for each batch,
        in the same manner as the `mz_array` and `intensity_array` columns.
        Each value in these columns must be a list of numbers. Missing
        columns are ignored.
    shuffle : bool, optional
        Shuffle the spectra in a new order each epoch. Blocks of `batch_size`
        consecutive spectra are read in a random order, then the spectra are
        shuffled within a buffer of 16 blocks. Shuffling cannot be combined
        with the `filter`, `sampler`, `samples`, `shard_granularity`, or
        `with_row_id` options of the `LanceDataset`.
    seed : int, optional
        The random seed for shuffling. If `None`, PyTorch's initial seed
        (`torch.initial_seed()`) is used. In distributed training, every
        process must use the same seed.
    **kwargs : dict
        Keyword arguments to initialize a
        `[lance.torch.data.LanceDataset](https://github.com/lance-format/lance/blob/92aa361099f42a40e9aa9f9915d041fe1dd30671/python/python/lance/torch/data.py#L177)`.

    Attributes
    ----------
    peak_files : list of str
    peak_file_hashes : list of str
    path : Path
    n_spectra : int
    dataset : lance.LanceDataset
    tokenizer : PeptideTokenizer
        The tokenizer for the annotations.
    annotations : str
        The annotation column in the dataset.

    """

    def __init__(
        self,
        spectra: pl.DataFrame | PathLike | Iterable[PathLike],
        annotations: str,
        tokenizer: PeptideTokenizer,
        batch_size: int,
        path: PathLike = None,
        parse_kwargs: dict | None = None,
        pad_fields: str | Iterable[str] | None = None,
        overwrite: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
        **kwargs: dict,
    ) -> None:
        """Initialize an AnnotatedSpectrumDataset."""
        self.tokenizer = tokenizer
        self.annotations = annotations
        super().__init__(
            spectra=spectra,
            batch_size=batch_size,
            path=path,
            parse_kwargs=parse_kwargs,
            pad_fields=pad_fields,
            overwrite=overwrite,
            shuffle=shuffle,
            seed=seed,
            **kwargs,
        )

    def _to_tensor(
        self,
        batch: pa.RecordBatch,
        **ignored: dict[Any],
    ) -> dict[str, torch.Tensor | list[str | torch.Tensor]]:
        """Convert a record batch to tensors.

        Parameters
        ----------
        batch : pyarrow.RecordBatch
            The batch of data.
        **ignored : dict[Any]
            Ignored keyword arguments to maintain compatibility with
            pylance.

        Returns
        -------
        dict of str to tensors or lists
            The batch of data as a Python dict.

        """
        batch = super()._to_tensor(batch)
        batch[self.annotations] = self.tokenizer.tokenize(
            batch[self.annotations],
            add_start=self.tokenizer.start_token is not None,
            add_stop=self.tokenizer.stop_token is not None,
        )
        return batch

    @classmethod
    def from_lance(
        cls,
        path: PathLike,
        annotations: str,
        tokenizer: PeptideTokenizer,
        batch_size: int,
        parse_kwargs: dict | None = None,
        pad_fields: str | Iterable[str] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        **kwargs: dict,
    ) -> AnnotatedSpectrumDataset:
        """Load a previously created lance dataset.

        Parameters
        ----------
        path : PathLike
            The path of the lance dataset.
        annotations : str
            The column name containing the annotations.
        tokenizer : PeptideTokenizer
            The tokenizer used to transform the annotations into PyTorch
            tensors.
        batch_size : int
            The batch size to use for loading mass spectra. Note that this is
            independent from the batch size for the PyTorch DataLoader.
        parse_kwargs : dict, optional
            Keyword arguments passed `depthcharge.spectra_to_stream()` for
            peak files that are provided.
        pad_fields : str or iterable of str, optional
            Additional list columns to pad into a single tensor for each batch,
            in the same manner as the `mz_array` and `intensity_array` columns.
            Each value in these columns must be a list of numbers. Missing
            columns are ignored.
        shuffle : bool, optional
            Shuffle the spectra in a new order each epoch. Blocks of
            `batch_size` consecutive spectra are read in a random order, then
            the spectra are shuffled within a buffer of 16 blocks. Shuffling
            cannot be combined with the `filter`, `sampler`, `samples`,
            `shard_granularity`, or `with_row_id` options of the
            `LanceDataset`.
        seed : int, optional
            The random seed for shuffling. If `None`, PyTorch's initial seed
            (`torch.initial_seed()`) is used. In distributed training, every
            process must use the same seed.
        **kwargs : dict
            Keyword arguments to initialize a
            `[lance.torch.data.LanceDataset](https://github.com/lance-format/lance/blob/92aa361099f42a40e9aa9f9915d041fe1dd30671/python/python/lance/torch/data.py#L177)`.

        Returns
        -------
        AnnotatedSpectrumDataset
            The dataset of annotated mass spectra.

        """
        return cls(
            spectra=None,
            annotations=annotations,
            tokenizer=tokenizer,
            batch_size=batch_size,
            path=path,
            parse_kwargs=parse_kwargs,
            pad_fields=pad_fields,
            shuffle=shuffle,
            seed=seed,
            **kwargs,
        )


class StreamingSpectrumDataset(IterableDataset):
    """Stream mass spectra from a file or DataFrame.

    While the on-disk dataset provided by `depthcharge.data.SpectrumDataset`
    provides an excellent option for model training, this class provides
    a PyTorch Dataset that is more suitable for inference.

    When using a `StreamingSpectrumDataset`, the order of mass spectra
    cannot be shuffled.

    The `batch_size` parameter for this class independent of the `batch_size`
    of the PyTorch DataLoader. Generally, we only want the former parameter to
    greater than 1. Additionally, this dataset should not be
    used with a DataLoader set to `num_workers` > 1, unless specific care is
    used to handle the
    [caveats of a PyTorch IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)

    Parameters
    ----------
    spectra : polars.DataFrame, PathLike, or list of PathLike
        Spectra to add to this collection. These may be a DataFrame parsed
        with `depthcharge.spectra_to_df()`, parquet files created with
        `depthcharge.spectra_to_parquet()`, or a peak file in the mzML,
        mzXML, MGF, or Bruker TDF format.
    batch_size : int
        The batch size to use for loading mass spectra. Note that this is
        independent from the batch size for the PyTorch DataLoader.
    pad_fields : str or iterable of str, optional
        Additional list columns to pad into a single tensor for each batch,
        in the same manner as the `mz_array` and `intensity_array` columns.
        Each value in these columns must be a list of numbers. Missing
        columns are ignored.
    **parse_kwargs : dict
        Keyword arguments passed `depthcharge.spectra_to_stream()` for
        peak files that are provided. This argument has no effect for
        DataFrame or parquet file inputs.

    Attributes
    ----------
    batch_size : int
        The batch size to use for loading mass spectra.

    """

    def __init__(
        self,
        spectra: pl.DataFrame | PathLike | Iterable[PathLike],
        batch_size: int,
        pad_fields: str | Iterable[str] | None = None,
        **parse_kwargs: dict,
    ) -> None:
        """Initialize a StreamingSpectrumDataset."""
        super().__init__()
        self.batch_size = batch_size
        self._pad_fields = _get_pad_fields(pad_fields)
        self._spectra = utils.listify(spectra)
        self._parse_kwargs = parse_kwargs

    def __iter__(self) -> dict[str, Any]:
        """Yield a batch mass spectra."""
        records = _get_records(
            self._spectra,
            batch_size=self.batch_size,
            **self._parse_kwargs,
        )
        for batch in records:
            yield _to_tensor(batch, self._pad_fields)


class _SpectrumSampler(Sampler):
    """Read batches of spectra, optionally in a random order.

    Batches are divided among DataLoader workers and distributed training
    processes when iteration starts, so that each spectrum is read once.

    Parameters
    ----------
    shuffle : bool
        Shuffle the spectra.
    seed : int
        The random seed for shuffling.
    buffer_size : int, optional
        The number of batches to shuffle spectra between.

    """

    def __init__(
        self,
        shuffle: bool,
        seed: int,
        buffer_size: int = 16,
    ) -> None:
        """Initialize the sampler."""
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch.

        Parameters
        ----------
        epoch : int
            The epoch number.

        """
        self.epoch = epoch

    def __call__(
        self,
        dataset: lance.LanceDataset,
        *args: tuple,
        batch_size: int = 128,
        columns: list[str] | dict[str, str] | None = None,
        batch_readahead: int = 16,
        **kwargs: dict,
    ) -> Generator[pa.RecordBatch]:
        """Yield batches of spectra.

        Parameters
        ----------
        dataset : lance.LanceDataset
            The dataset to read.
        *args : tuple
            Ignored.
        batch_size : int, optional
            The number of spectra in each batch.
        columns : list of str or dict of str to str, optional
            The columns to read.
        batch_readahead : int, optional
            The number of batches to read ahead.
        **kwargs : dict
            Ignored.

        Yields
        ------
        pyarrow.RecordBatch
            A batch of spectra.

        """
        rank, world_size = get_global_rank(), get_global_world_size()
        n_rows = dataset.count_rows()
        blocks = [
            (start, min(start + batch_size, n_rows))
            for start in range(0, n_rows, batch_size)
        ]

        rng = np.random.default_rng([self.seed, self.epoch, _shared_seed()])
        self.epoch += 1
        if self.shuffle:
            blocks = [blocks[i] for i in rng.permutation(len(blocks))]

        blocks = blocks[rank::world_size]
        if not blocks:
            return

        # Lance's ShardedBatchSampler reads row ranges in the same way:
        stream = dataset._ds.take_scan(
            blocks,
            columns=columns,
            batch_readahead=batch_readahead,
        )

        if not self.shuffle:
            yield from stream
            return

        buffer = []
        for batch in stream:
            buffer.append(batch)
            if len(buffer) == self.buffer_size:
                yield from _shuffle_batches(buffer, rng, batch_size)
                buffer = []

        if buffer:
            yield from _shuffle_batches(buffer, rng, batch_size)


def _shuffle_batches(
    batches: list[pa.RecordBatch],
    rng: np.random.Generator,
    batch_size: int,
) -> list[pa.RecordBatch]:
    """Shuffle the rows between record batches.

    Parameters
    ----------
    batches : list of pyarrow.RecordBatch
        The record batches to shuffle.
    rng : numpy.random.Generator
        The random number generator.
    batch_size : int
        The maximum number of rows in each output batch.

    Returns
    -------
    list of pyarrow.RecordBatch
        The shuffled record batches.

    """
    table = pa.Table.from_batches(batches)
    table = table.take(rng.permutation(table.num_rows))
    return table.combine_chunks().to_batches(batch_size)


def _shared_seed() -> int:
    """Get a seed that is shared by the DataLoader workers for an epoch.

    The DataLoader chooses a new base seed for its workers each epoch,
    unless they are persistent. However, the base seed differs between
    distributed training processes, so it is not used in that case.

    Returns
    -------
    int
        The base seed of the DataLoader workers, or 0 if not in a DataLoader
        worker or during distributed training.

    """
    info = torch.utils.data.get_worker_info()
    if info is None or get_dist_world_size() > 1:
        return 0

    return info.seed - info.id


def _get_sampler(
    shuffle: bool,
    seed: int | None,
    kwargs: dict,
) -> _SpectrumSampler | None:
    """Get the sampler for a dataset.

    Parameters
    ----------
    shuffle : bool
        Shuffle the spectra.
    seed : int or None
        The random seed for shuffling.
    kwargs : dict
        The keyword arguments for the LanceDataset.

    Returns
    -------
    _SpectrumSampler or None
        The sampler, or `None` if the LanceDataset options require one
        of Lance's samplers.

    Raises
    ------
    ValueError
        Raised if shuffling is combined with incompatible options.

    """
    incompatible = [
        key
        for key in [
            "filter",
            "sampler",
            "samples",
            "shard_granularity",
            "with_row_id",
            "rank",
            "world_size",
        ]
        if kwargs.get(key)
    ]
    if incompatible:
        if shuffle:
            raise ValueError(
                "Shuffling cannot be combined with the following options: "
                f"{', '.join(incompatible)}."
            )

        return None

    seed = torch.initial_seed() if seed is None else seed
    return _SpectrumSampler(shuffle=shuffle, seed=seed)


def _get_records(
    data: list[pl.DataFrame | PathLike], **kwargs: dict
) -> Generator[pa.RecordBatch]:
    """Yields RecordBatches for data.

    Parameters
    ----------
    data : list of polars.DataFrame or PathLike
        The data to add.
    **kwargs : dict
        Keyword arguments for the parser. If present, `batch_size` is also
        used for DataFrame and parquet inputs.

    Yields
    ------
    pyarrow.RecordBatch
        The batches of spectra.

    """
    batch_size = kwargs.get("batch_size")
    parquet_kwargs = {} if batch_size is None else {"batch_size": batch_size}
    for spectra in data:
        try:
            spectra = (
                spectra.lazy()
                .collect()
                .rechunk()
                .to_arrow()
                .to_batches(max_chunksize=batch_size)
            )
        except AttributeError:
            try:
                spectra = pq.ParquetFile(spectra).iter_batches(
                    **parquet_kwargs
                )
            except (pa.ArrowInvalid, TypeError, OSError):
                spectra = arrow.spectra_to_stream(spectra, **kwargs)

        yield from spectra


def _filter_duplicates(
    data: list[pl.DataFrame | PathLike],
    existing: Iterable[str] = (),
    warn: bool = True,
) -> list[pl.DataFrame | PathLike]:
    """Remove peak files that have already been added.

    Parameters
    ----------
    data : list of polars.DataFrame or PathLike
        The data to add.
    existing : iterable of str, optional
        The fingerprints of peak files that have already been added.
    warn : bool, optional
        Warn about skipped peak files. Otherwise, they are logged.

    Returns
    -------
    list of polars.DataFrame or PathLike
        The data to add, without duplicate peak files. DataFrame and
        parquet inputs are always kept.

    """
    seen = set(existing)
    keep = []
    skipped = []
    for spectra in data:
        if _is_peak_file(spectra):
            peak_file_hash = hash_peak_file(spectra)
            if peak_file_hash in seen:
                skipped.append(AnyPath(spectra).name)
                continue

            seen.add(peak_file_hash)

        keep.append(spectra)

    if skipped:
        msg = (
            f"Skipped {len(skipped)} peak file(s) that were already added to "
            f"the dataset: {', '.join(skipped)}"
        )
        if warn:
            warnings.warn(msg)
        else:
            LOGGER.info(msg)

    return keep


def _append(
    dataset: lance.LanceDataset,
    data: list[pl.DataFrame | PathLike],
    parse_kwargs: dict,
) -> lance.LanceDataset:
    """Append spectra to an existing lance dataset.

    The new spectra are cast to the schema of the existing dataset. For
    example, datasets created from a polars DataFrame use large string
    and list types, whereas parsed peak files do not.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The existing lance dataset.
    data : list of polars.DataFrame or PathLike
        The data to add.
    parse_kwargs : dict
        The keyword arguments for parsing peak files.

    Returns
    -------
    lance.LanceDataset
        The updated lance dataset.

    """
    schema = dataset.schema
    records = _cast_records(_get_records(data, **parse_kwargs), schema)

    # Check the first batch here, so errors aren't wrapped by lance:
    first = next(records, None)
    if first is None:
        return dataset

    return lance.write_dataset(
        pa.RecordBatchReader.from_batches(
            schema,
            itertools.chain([first], records),
        ),
        dataset.uri,
        mode="append",
    )


def _cast_records(
    records: Iterable[pa.RecordBatch],
    schema: pa.Schema,
) -> Generator[pa.RecordBatch]:
    """Cast record batches to a schema.

    Parameters
    ----------
    records : iterable of pyarrow.RecordBatch
        The record batches to cast.
    schema : pyarrow.Schema
        The schema to cast to. The record batches must have the same columns,
        although they may be in a different order.

    Yields
    ------
    pyarrow.RecordBatch
        The record batches with the new schema.

    Raises
    ------
    ValueError
        Raised if the columns of a record batch do not match the schema.

    """
    for record in records:
        missing = set(schema.names) - set(record.schema.names)
        unexpected = set(record.schema.names) - set(schema.names)
        if missing or unexpected:
            raise ValueError(
                "The new spectra do not have the same columns as the "
                f"existing lance dataset. Missing: {sorted(missing)}. "
                f"Unexpected: {sorted(unexpected)}."
            )

        table = pa.Table.from_batches([record]).select(schema.names)
        yield from table.cast(schema).to_batches()


def _get_peak_file_hashes(dataset: lance.LanceDataset) -> list[str]:
    """Get the peak file fingerprints in a lance dataset.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The lance dataset.

    Returns
    -------
    list of str
        The unique peak file fingerprints, or an empty list if the
        dataset does not have a `peak_file_hash` column.

    """
    if "peak_file_hash" not in dataset.schema.names:
        return []

    return (
        dataset.to_table(columns=["peak_file_hash"])
        .column(0)
        .unique()
        .drop_null()
        .to_pylist()
    )


def _check_appendable(
    dataset: lance.LanceDataset,
    data: list[pl.DataFrame | PathLike],
    parse_kwargs: dict,
) -> None:
    """Verify that new peak files can be added to an existing dataset.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The existing lance dataset.
    data : list of polars.DataFrame or PathLike
        The data to add.
    parse_kwargs : dict
        The keyword arguments for parsing peak files.

    Raises
    ------
    ValueError
        Raised if the data cannot be added to the existing dataset.

    """
    hint = "Use `overwrite=True` to replace it."
    columns = set(dataset.schema.names)
    if "peak_file_hash" not in columns:
        raise ValueError(
            f"The lance dataset at '{dataset.uri}' was created with a "
            f"previous version of depthcharge. {hint}"
        )

    if not all(_is_peak_file(x) for x in data):
        raise ValueError(
            "DataFrame and parquet inputs cannot be checked against the "
            f"existing lance dataset at '{dataset.uri}'. {hint} Use "
            "`add_spectra()` to add them to the existing dataset."
        )

    custom_fields = parse_kwargs.get("custom_fields")
    custom_fields = [] if custom_fields is None else custom_fields
    missing = [
        f.name for f in utils.listify(custom_fields) if f.name not in columns
    ]
    if missing:
        raise ValueError(
            f"The lance dataset at '{dataset.uri}' is missing the custom "
            f"fields: {', '.join(missing)}. {hint}"
        )


def _is_peak_file(data: pl.DataFrame | PathLike) -> bool:
    """Determine whether data is a peak file.

    Parameters
    ----------
    data : polars.DataFrame or PathLike
        The data to check.

    Returns
    -------
    bool
        False if the data is a DataFrame or parquet file, True otherwise.

    """
    if isinstance(data, pl.DataFrame | pl.LazyFrame):
        return False

    try:
        pq.ParquetFile(data)
    except (pa.ArrowInvalid, TypeError, OSError):
        return True

    return False


def _get_pad_fields(pad_fields: str | Iterable[str] | None) -> tuple[str]:
    """Get the columns to pad in each batch.

    Parameters
    ----------
    pad_fields : str or iterable of str, optional
        Additional columns to pad.

    Returns
    -------
    tuple of str
        The columns to pad, including the mass spectrum arrays.

    """
    pad_fields = [] if pad_fields is None else utils.listify(pad_fields)
    return tuple(dict.fromkeys(["mz_array", "intensity_array", *pad_fields]))


def _to_tensor(
    batch: pa.RecordBatch,
    pad_fields: Iterable[str] = ("mz_array", "intensity_array"),
) -> dict[str, torch.Tensor | list[str | torch.Tensor]]:
    """Convert a record batch to tensors.

    Parameters
    ----------
    batch : pyarrow.RecordBatch
        The batch of data.
    pad_fields : iterable of str
        The columns to pad into a single tensor. Missing columns
        are ignored.

    Returns
    -------
    dict of str to tensors or lists
        The batch of data as a Python dict.

    """
    out = {}
    for name, column in zip(batch.schema.names, batch.columns):
        if isinstance(column, pa.ChunkedArray):
            column = column.combine_chunks()

        values = _column_to_tensor(column, pad=name in pad_fields)
        if values is None:
            if name in pad_fields:
                raise ValueError(
                    f"Cannot pad the '{name}' column. Padded columns must be "
                    "lists of numbers without missing values."
                )

            # Fall back to converting Python objects:
            values = _tensorize(column.to_pylist())

        out[name] = values

    return out


def _column_to_tensor(
    column: pa.Array,
    pad: bool = False,
) -> torch.Tensor | list[torch.Tensor] | None:
    """Convert a numeric Arrow column to tensors without Python objects.

    The resulting tensors have the same data types as those created by
    `torch.tensor()` from the equivalent Python objects.

    Parameters
    ----------
    column : pyarrow.Array
        The column to convert.
    pad : bool, optional
        Pad a list column into a single 2D tensor. Otherwise, rows of a list
        column with different lengths are returned as a list of 1D tensors.

    Returns
    -------
    torch.Tensor, list of torch.Tensor, or None
        The converted column, or `None` if the column is not a numeric
        or list of numeric column without missing values.

    """
    if not len(column) or column.null_count:
        return None

    is_list = pa.types.is_list(column.type) or pa.types.is_large_list(
        column.type
    )

    if is_list:
        lengths = torch.from_numpy(np.diff(column.offsets.to_numpy()))
        column = column.flatten()
        if column.null_count:
            return None
    elif pad:
        return None

    dtype = _torch_dtype(column.type)
    if dtype is None:
        return None

    # Copy, because Arrow memory is read-only:
    values = torch.tensor(column.to_numpy(zero_copy_only=False), dtype=dtype)
    if not is_list:
        return values

    max_length = int(lengths.max())
    if bool((lengths == max_length).all()):
        return values.reshape(len(lengths), max_length)

    if not pad:
        return list(values.split(lengths.tolist()))

    mask = torch.arange(max_length) < lengths[:, None]
    padded = torch.zeros((len(lengths), max_length), dtype=dtype)
    padded[mask] = values
    return padded


def _torch_dtype(arrow_type: pa.DataType) -> torch.dtype | None:
    """Get the PyTorch data type for an Arrow data type.

    Parameters
    ----------
    arrow_type : pyarrow.DataType
        The Arrow data type.

    Returns
    -------
    torch.dtype or None
        The data type that `torch.tensor()` uses for the equivalent Python
        objects, or `None` if the Arrow data type is not numeric.

    """
    if pa.types.is_floating(arrow_type):
        return torch.get_default_dtype()

    if pa.types.is_integer(arrow_type):
        return torch.int64

    if pa.types.is_boolean(arrow_type):
        return torch.bool

    return None


def _tensorize(obj: Any) -> Any:  # noqa: ANN401
    """Turn lists into tensors.

    Parameters
    ----------
    obj : any object
        If a list, attempt to make a tensor. If not or if it fails,
        return the obj unchanged.

    Returns
    -------
    Any
        Whatever type the object is, unless its been transformed to
        a PyTorch tensor.

    """
    if not isinstance(obj, list):
        return obj

    try:
        return torch.tensor(obj)
    except (ValueError, RuntimeError):
        obj = [_tensorize(x) for x in obj]

    return obj
