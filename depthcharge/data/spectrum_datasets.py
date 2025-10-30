"""Serve mass spectra to neural networks."""

from __future__ import annotations

import copy
import logging
import uuid
from collections.abc import Generator, Iterable
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import lance
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from cloudpathlib import AnyPath
from lance.torch.data import LanceDataset
from torch import nn
from torch.utils.data import IterableDataset

from .. import utils
from ..tokenizers import PeptideTokenizer
from . import arrow

LOGGER = logging.getLogger(__name__)


class SpectrumDataset(LanceDataset):
    """Store and access a collection of mass spectra.

    Parse and/or add mass spectra to an index in the
    [lance data format](https://lancedb.github.io/lance/index.html).
    This format enables fast random access to spectra for training.
    This file is then served as a PyTorch IterableDataset, allowing
    spectra to be accessed efficiently for training and inference.
    This is accomplished using the
    [Lance PyTorch integration](https://lancedb.github.io/lance/integrations/pytorch.html).

    The `batch_size` parameter for this class indepedent of the `batch_size`
    of the PyTorch DataLoader. Generally, we only want the former parameter to
    greater than 1. Additionally, this dataset should not be
    used with a DataLoader set to `max_workers` > 1, unless specific care is
    used to handle the
    [caveats of a PyTorch IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)

    If you wish to use an existing lance dataset, use the `from_lance()`
    method.

    Parameters
    ----------
    spectra : polars.DataFrame, PathLike, or list of PathLike
        Spectra to add to this collection. These may be a DataFrame parsed
        with `depthcharge.spectra_to_df()`, parquet files created with
        `depthcharge.spectra_to_parquet()`, or a peak file in the mzML,
        mzXML, or MGF format. Additional spectra can be added later using
        the `.add_spectra()` method.
    batch_size : int
        The batch size to use for loading mass spectra. Note that this is
        independent from the batch size for the PyTorch DataLoader.
    path : PathLike, optional.
        The name and path of the lance dataset. If the path does
        not contain the `.lance` then it will be added.
        If `None`, a file will be created in a temporary directory.
    parse_kwargs : dict, optional
        Keyword arguments passed `depthcharge.spectra_to_stream()` for
        peak files that are provided. This argument has no affect for
        DataFrame or parquet file inputs.
    **kwargs : dict
        Keyword arguments to initialize a
        `[lance.torch.data.LanceDataset](https://lancedb.github.io/lance/api/python/lance.torch.html#lance.torch.data.LanceDataset)`.

    Attributes
    ----------
    peak_files : list of str
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
        **kwargs: dict,
    ) -> None:
        """Initialize a SpectrumDataset."""
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

        super().__init__(dataset, batch_size, **kwargs)

    def add_spectra(
        self,
        spectra: pl.DataFrame | PathLike | Iterable[PathLike],
    ) -> SpectrumDataset:
        """Add mass spectrometry data to the lance dataset.

        Note that depthcharge does not verify whether the provided spectra
        already exist in the lance dataset.

        Parameters
        ----------
        spectra : polars.DataFrame, PathLike, or list of PathLike
            Spectra to add to this collection. These may be a DataFrame parsed
            with `depthcharge.spectra_to_df()`, parquet files created with
            `depthcharge.spectra_to_parquet()`, or a peak file in the mzML,
            mzXML, or MGF format.

        """
        spectra = utils.listify(spectra)
        batch = next(_get_records(spectra, **self._init_kwargs))
        self.dataset = lance.write_dataset(
            _get_records(spectra, **self._parse_kwargs),
            self._path,
            mode="append",
            schema=batch.schema,
        )

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
    def path(self) -> Path:
        """The path to the underyling lance dataset."""
        return self._path

    @classmethod
    def from_lance(
        cls,
        path: PathLike,
        batch_size: int,
        parse_kwargs: dict | None = None,
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
        **kwargs : dict
            Keyword arguments to initialize a
            `[lance.torch.data.LanceDataset](https://lancedb.github.io/lance/api/python/lance.torch.html#lance.torch.data.LanceDataset)`.

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
        batch = {k: _tensorize(v) for k, v in batch.to_pydict().items()}
        batch["mz_array"] = nn.utils.rnn.pad_sequence(
            batch["mz_array"],
            batch_first=True,
        )
        batch["intensity_array"] = nn.utils.rnn.pad_sequence(
            batch["intensity_array"],
            batch_first=True,
        )
        return batch


class AnnotatedSpectrumDataset(SpectrumDataset):
    """Store and access a collection of annotated mass spectra.

    Parse and/or add mass spectra to an index in the
    [lance data format](https://lancedb.github.io/lance/index.html).
    This format enables fast random access to spectra for training.
    This file is then served as a PyTorch IterableDataset, allowing
    spectra to be accessed efficiently for training and inference.
    This is accomplished using the
    [Lance PyTorch integration](https://lancedb.github.io/lance/integrations/pytorch.html).

    The `batch_size` parameter for this class indepedent of the `batch_size`
    of the PyTorch DataLoader. Generally, we only want the former parameter to
    greater than 1. Additionally, this dataset should not be
    used with a DataLoader set to `max_workers` > 1, unless specific care is
    used to handle the
    [caveats of a PyTorch IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)

    If you wish to use an existing lance dataset, use the `from_lance()`
    method.

    Parameters
    ----------
    spectra : polars.DataFrame, PathLike, or list of PathLike
        Spectra to add to this collection. These may be a DataFrame parsed
        with `depthcharge.spectra_to_df()`, parquet files created with
        `depthcharge.spectra_to_parquet()`, or a peak file in the mzML,
        mzXML, or MGF format. Additional spectra can be added later using
        the `.add_spectra()` method.
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
    parse_kwargs : dict, optional
        Keyword arguments passed `depthcharge.spectra_to_stream()` for
        peak files that are provided. This argument has no affect for
        DataFrame or parquet file inputs.
    **kwargs : dict
        Keyword arguments to initialize a
        `[lance.torch.data.LanceDataset](https://lancedb.github.io/lance/api/python/lance.torch.html#lance.torch.data.LanceDataset)`.

    Attributes
    ----------
    peak_files : list of str
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
        **kwargs : dict
            Keyword arguments to initialize a
            `[lance.torch.data.LanceDataset](https://lancedb.github.io/lance/api/python/lance.torch.html#lance.torch.data.LanceDataset)`.

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
            **kwargs,
        )


class StreamingSpectrumDataset(IterableDataset):
    """Stream mass spectra from a file or DataFrame.

    While the on-disk dataset provided by `depthcharge.data.SpectrumDataset`
    provides an excellent option for model training, this class provides
    a PyTorch Dataset that is more suitable for inference.

    When using a `StreamingSpectrumDataset`, the order of mass spectra
    cannot be shuffled.

    The `batch_size` parameter for this class indepedent of the `batch_size`
    of the PyTorch DataLoader. Generally, we only want the former parameter to
    greater than 1. Additionally, this dataset should not be
    used with a DataLoader set to `max_workers` > 1, unless specific care is
    used to handle the
    [caveats of a PyTorch IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)

    Parameters
    ----------
    spectra : polars.DataFrame, PathLike, or list of PathLike
        Spectra to add to this collection. These may be a DataFrame parsed
        with `depthcharge.spectra_to_df()`, parquet files created with
        `depthcharge.spectra_to_parquet()`, or a peak file in the mzML,
        mzXML, or MGF format.
    batch_size : int
        The batch size to use for loading mass spectra. Note that this is
        independent from the batch size for the PyTorch DataLoader.
    **parse_kwargs : dict
        Keyword arguments passed `depthcharge.spectra_to_stream()` for
        peak files that are provided. This argument has no affect for
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
        **parse_kwargs: dict,
    ) -> None:
        """Initialize a StreamingSpectrumDataset."""
        super().__init__()
        self.batch_size = batch_size
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
            yield _to_tensor(batch)


def _get_records(
    data: list[pl.DataFrame | PathLike], **kwargs: dict
) -> Generator[pa.RecordBatch]:
    """Yields RecordBatches for data.

    Parameters
    ----------
    data : list of polars.DataFrame or PathLike
        The data to add.
    **kwargs : dict
        Keyword arguments for the parser.

    """
    for spectra in data:
        try:
            spectra = spectra.lazy().collect().to_arrow().to_batches()
        except AttributeError:
            try:
                spectra = pq.ParquetFile(spectra).iter_batches()
            except (pa.ArrowInvalid, TypeError):
                spectra = arrow.spectra_to_stream(spectra, **kwargs)

        yield from spectra


def _to_tensor(
    batch: pa.RecordBatch,
) -> dict[str, torch.Tensor | list[str | torch.Tensor]]:
    """Convert a record batch to tensors.

    Parameters
    ----------
    batch : pyarrow.RecordBatch
        The batch of data.

    Returns
    -------
    dict of str to tensors or lists
        The batch of data as a Python dict.

    """
    batch = {k: _tensorize(v) for k, v in batch.to_pydict().items()}
    batch["mz_array"] = nn.utils.rnn.pad_sequence(
        batch["mz_array"],
        batch_first=True,
    )

    batch["intensity_array"] = nn.utils.rnn.pad_sequence(
        batch["intensity_array"],
        batch_first=True,
    )
    return batch


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
    except ValueError:
        obj = [_tensorize(x) for x in obj]

    return obj
