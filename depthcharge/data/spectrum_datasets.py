"""Serve mass spectra to neural networks."""

from __future__ import annotations

import logging
import uuid
import warnings
from collections.abc import Generator, Iterable
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import lance
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from cloudpathlib import AnyPath
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .. import utils
from ..tokenizers import PeptideTokenizer
from . import arrow

LOGGER = logging.getLogger(__name__)


class CollateFnMixin:
    """Define a common collate function."""

    def __init__(self) -> None:
        """Specify the float precision."""
        self.precision = torch.float64

    def collate_fn(
        self,
        batch: Iterable[dict[Any]],
    ) -> dict[str, torch.Tensor | list[Any]]:
        """The collate function for a SpectrumDataset.

        Transform compatible data types into PyTorch tensors and
        pad the m/z and intensities arrays of each mass spectrum with
        zeros to be stacked into tensor.

        Parameters
        ----------
        batch : iterable of dict
            A batch of data.

        Returns
        -------
        dict of str, tensor or list
            A dictionary mapping the columns of the lance dataset
            to a PyTorch tensor or list of values.

        """
        mz_array = nn.utils.rnn.pad_sequence(
            [s.pop("mz_array") for s in batch],
            batch_first=True,
        )

        intensity_array = nn.utils.rnn.pad_sequence(
            [s.pop("intensity_array") for s in batch],
            batch_first=True,
        )

        batch = torch.utils.data.default_collate(batch)
        batch["mz_array"] = mz_array
        batch["intensity_array"] = intensity_array

        for key, val in batch.items():
            if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
                batch[key] = val.type(self.precision)

        return batch

    def loader(
        self,
        precision: Literal[
            torch.float16, torch.float32, torch.float64
        ] = torch.float64,
        **kwargs: dict,
    ) -> DataLoader:
        """Create a suitable PyTorch DataLoader."""
        self.precision = precision
        if kwargs.get("collate_fn", False):
            warnings.warn("The default collate_fn was overridden.")
        else:
            kwargs["collate_fn"] = self.collate_fn

        return DataLoader(self, **kwargs)


class SpectrumDataset(Dataset, CollateFnMixin):
    """Store and access a collection of mass spectra.

    Parse and/or add mass spectra to an index in the
    [lance data format](https://lancedb.github.io/lance/index.html).
    This format enables fast random access to spectra for training.
    This file is then served as a PyTorch Dataset, allowing spectra
    to be accessed efficiently for training and inference.

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
    path : PathLike, optional.
        The name and path of the lance dataset. If the path does
        not contain the `.lance` then it will be added.
        If `None`, a file will be created in a temporary directory.
    **kwargs : dict
        Keyword arguments passed `depthcharge.spectra_to_stream()` for
        peak files that are provided. This argument has no affect for
        DataFrame or parquet file inputs.

    Attributes
    ----------
    peak_files : list of str
    path : Path

    """

    def __init__(
        self,
        spectra: pl.DataFrame | PathLike | Iterable[PathLike],
        path: PathLike | None = None,
        **kwargs: dict,
    ) -> None:
        """Initialize a SpectrumDataset."""
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
            batch = next(_get_records(spectra, **kwargs))
            lance.write_dataset(
                _get_records(spectra, **kwargs),
                str(self._path),
                mode="overwrite",
                schema=batch.schema,
            )

        elif not self._path.exists():
            raise ValueError("No spectra were provided")

        self._dataset = lance.dataset(str(self._path))

    def add_spectra(
        self,
        spectra: pl.DataFrame | PathLike | Iterable[PathLike],
        **kwargs: dict,
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
        **kwargs : dict
            Keyword arguments passed `depthcharge.spectra_to_stream()` for
            peak files that are provided. This argument has no affect for
            DataFrame or parquet file inputs.

        """
        spectra = utils.listify(spectra)
        batch = next(_get_records(spectra, **kwargs))
        self._dataset = lance.write_dataset(
            _get_records(spectra, **kwargs),
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
        return {
            k: _tensorize(v[0])
            for k, v in self._dataset.take([idx]).to_pydict().items()
        }

    def __len__(self) -> int:
        """The number of spectra in the lance dataset."""
        return self._dataset.count_rows()

    def __del__(self) -> None:
        """Cleanup the temporary directory."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    @property
    def peak_files(self) -> list[str]:
        """The files currently in the lance dataset."""
        return (
            self._dataset.to_table(columns=["peak_file"])
            .column(0)
            .unique()
            .to_pylist()
        )

    @property
    def path(self) -> Path:
        """The path to the underyling lance dataset."""
        return self._path

    @classmethod
    def from_lance(cls, path: PathLike, **kwargs: dict) -> SpectrumDataset:
        """Load a previously created lance dataset.

        Parameters
        ----------
        path : PathLike
            The path of the lance dataset.
        **kwargs : dict
            Keyword arguments passed `depthcharge.spectra_to_stream()` for
            peak files that are added. This argument has no affect for
            DataFrame or parquet file inputs.

        """
        return cls(spectra=None, path=path, **kwargs)


class AnnotatedSpectrumDataset(SpectrumDataset):
    """Store and access a collection of annotated mass spectra.

    Parse and/or add mass spectra to an index in the
    [lance data format](https://lancedb.github.io/lance/index.html).
    This format enables fast random access to spectra for training.
    This file is then served as a PyTorch Dataset, allowing spectra
    to be accessed efficiently for training and inference.

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
    path : PathLike, optional.
        The name and path of the lance dataset. If the path does
        not contain the `.lance` then it will be added.
        If ``None``, a file will be created in a temporary directory.
    **kwargs : dict
        Keyword arguments passed `depthcharge.spectra_to_stream()` for
        peak files that are provided. This argument has no affect for
        DataFrame or parquet file inputs.

    Attributes
    ----------
    peak_files : list of str
    path : Path
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
        path: PathLike = None,
        **kwargs: dict,
    ) -> None:
        """Initialize an AnnotatedSpectrumDataset."""
        self.tokenizer = tokenizer
        self.annotations = annotations
        super().__init__(
            spectra=spectra,
            path=path,
            **kwargs,
        )

    def collate_fn(
        self,
        batch: Iterable[dict[Any]],
    ) -> dict[Any]:
        """The collate function for a AnnotatedSpectrumDataset.

        The mass spectra must be padded so that they fit nicely as a tensor.
        However, the padded elements are ignored during the subsequent steps.

        Parameters
        ----------
        batch : tuple of tuple of torch.Tensor
            A batch of data from an AnnotatedSpectrumDataset.

        Returns
        -------
        dict of str, tensor or list
            A dictionary mapping the columns of the lance dataset
            to a PyTorch tensor or list of values.

        """
        batch = super().collate_fn(batch)
        batch[self.annotations] = self.tokenizer.tokenize(
            batch[self.annotations]
        )
        return batch

    @classmethod
    def from_lance(
        cls,
        path: PathLike,
        annotations: str,
        tokenizer: PeptideTokenizer,
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
        **kwargs : dict
            Keyword arguments passed `depthcharge.spectra_to_stream()` for
            peak files that are added. This argument has no affect for
            DataFrame or parquet file inputs.

        """
        return cls(
            spectra=None,
            annotations=annotations,
            tokenizer=tokenizer,
            path=path,
            **kwargs,
        )


class StreamingSpectrumDataset(IterableDataset, CollateFnMixin):
    """Stream mass spectra from a file or DataFrame.

    While the on-disk dataset provided by `depthcharge.data.SpectrumDataset`
    provides an excellent option for model training, this class provides
    a PyTorch Dataset that is more suitable for inference.

    When using a `StreamingSpectrumDataset`, the order of mass spectra
    cannot be shuffled.

    The `batch_size` parameter for this class indepedent of the `batch_size`
    of the PyTorch DataLoader. The former specified how many spectra are
    simultaneously read into memory, whereas the latter specifies the batch
    size served to a model. Additionally, this dataset should not be
    used with a DataLoader set to `max_workers` > 1.

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
    **kwargs : dict
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
        **kwargs: dict,
    ) -> None:
        """Initialize a StreamingSpectrumDataset."""
        self.batch_size = batch_size
        self._spectra = utils.listify(spectra)
        self._kwargs = kwargs

    def __iter__(self) -> dict[str, Any]:
        """Yield a batch mass spectra."""
        records = _get_records(
            self._spectra,
            batch_size=self.batch_size,
            **self._kwargs,
        )
        for batch in records:
            for row in batch.to_pylist():
                yield {k: _tensorize(v) for k, v in row.items()}

    def loader(self, **kwargs: dict) -> DataLoader:
        """Create a suitable PyTorch DataLoader."""
        if kwargs.get("num_workers", 0) > 1:
            warnings.warn("'num_workers' > 1 may have unexpected behavior.")

        return super().loader(**kwargs)


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
        pass

    return obj
