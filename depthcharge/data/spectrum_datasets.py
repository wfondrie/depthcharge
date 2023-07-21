"""Parse mass spectra into an HDF5 file format."""
from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import dill
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .. import utils
from ..primitives import MassSpectrum
from ..tokenizers import PeptideTokenizer
from . import preprocessing
from .parsers import MgfParser, MzmlParser, MzxmlParser

LOGGER = logging.getLogger(__name__)


class SpectrumDataset(Dataset):
    """Store and access a collection of mass spectra.

    Parses one or more MS data files, adding the mass spectra to an HDF5
    file that indexes the mass spectra from one or multiple files.
    This allows depthcharge to access random mass spectra from many different
    files quickly and without loading them all into memory.

    Parameters
    ----------
    ms_data_files : PathLike or list of PathLike, optional
        The mzML, mzXML, or MGF files to include in this collection. Files
        can be added later using the ``.add_file()`` method.
    ms_level : int, optional
        The level of tandem mass spectra to use.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra. ``None``,
        the default, filters for the top 200 peaks above m/z 140,
        square root transforms the intensities and scales them to unit norm.
        See the preprocessing module for details and additional options.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    index_path : PathLike, optional.
        The name and path of the HDF5 file index. If the path does
        not contain the `.h5` or `.hdf5` extension, `.hdf5` will be added.
        If ``None``, a file will be created in a temporary directory.
    overwrite : bool, optional
        Overwrite previously indexed files? If ``False`` and new files are
        provided, they will be appended to the collection.

    Attributes
    ----------
    ms_files : list of str
    path : Path
    ms_level : int
    valid_charge : Optional[Iterable[int]]
    annotated : bool
    overwrite : bool
    n_spectra : int
    n_peaks : int
    """

    _annotated = False

    def __init__(
        self,
        ms_data_files: PathLike | Iterable[PathLike] = None,
        ms_level: int = 2,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        index_path: PathLike | None = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize a SpectrumIndex."""
        self._tmpdir = None
        if index_path is None:
            # Create a random temporary file:
            self._tmpdir = TemporaryDirectory()
            index_path = Path(self._tmpdir.name) / f"{uuid.uuid4()}.hdf5"

        index_path = Path(index_path)
        if index_path.suffix not in [".h5", ".hdf5"]:
            index_path = Path(index_path)
            index_path = Path(str(index_path) + ".hdf5")

        # Set attributes and check parameters:
        self._path = index_path
        self._ms_level = utils.check_positive_int(ms_level, "ms_level")
        self._valid_charge = valid_charge
        self._overwrite = bool(overwrite)
        self._handle = None
        self._file_offsets = np.array([0])
        self._file_map = {}
        self._locs = {}
        self._offsets = None

        if preprocessing_fn is not None:
            self._preprocessing_fn = utils.listify(preprocessing_fn)
        else:
            self._preprocessing_fn = [
                preprocessing.set_mz_range(min_mz=140),
                preprocessing.filter_intensity(max_num_peaks=200),
                preprocessing.scale_intensity(scaling="root"),
                preprocessing.scale_to_unit_norm,
            ]

        # Create the file if it doesn't exist.
        if not self.path.exists() or self.overwrite:
            with h5py.File(self.path, "w") as index:
                index.attrs["ms_level"] = self.ms_level
                index.attrs["n_spectra"] = 0
                index.attrs["n_peaks"] = 0
                index.attrs["annotated"] = self.annotated
                index.attrs["preprocessing_fn"] = _hash_obj(
                    tuple(self.preprocessing_fn)
                )
        else:
            self._validate_index()

        # Now parse spectra.
        if ms_data_files is not None:
            ms_data_files = utils.listify(ms_data_files)
            for ms_file in ms_data_files:
                self.add_file(ms_file)

    def _reindex(self) -> None:
        """Update the file mappings and offsets."""
        offsets = [0]
        for idx in range(len(self._handle)):
            grp = self._handle[str(idx)]
            offsets.append(grp.attrs["n_spectra"])
            self._file_map[grp.attrs["path"]] = idx

        self._file_offsets = np.cumsum([0] + offsets)

        # Build a map of 1D indices to 2D locations:
        grp_idx = 0
        for lin_idx in range(offsets[-1]):
            grp_idx += lin_idx >= offsets[grp_idx + 1]
            row_idx = lin_idx - offsets[grp_idx]
            self._locs[lin_idx] = (grp_idx, row_idx)

        self._offsets = None
        _ = self.offsets  # Reinitialize the offsets.

    def _validate_index(self) -> None:
        """Validate that the index is appropriate for this dataset."""
        preproc_hash = _hash_obj(tuple(self.preprocessing_fn))
        with self:
            try:
                assert self._handle.attrs["ms_level"] == self.ms_level
                assert self._handle.attrs["preprocessing_fn"] == preproc_hash
                if self._annotated:
                    assert self._handle.attrs["annotated"]
            except (KeyError, AssertionError):
                raise ValueError(
                    f"'{self.path}' already exists, but was created with "
                    "incompatible parameters. Use 'overwrite=True' to "
                    "overwrite it."
                )

            self._reindex()

    def _get_parser(
        self,
        ms_data_file: PathLike,
    ) -> MzmlParser | MzxmlParser | MgfParser:
        """Get the parser for the MS data file.

        Parameters
        ----------
        ms_data_file : PathLike
            The mass spectrometry data file to be parsed.

        Returns
        -------
        MzmlParser, MzxmlParser, or MgfParser
            The appropriate parser for the file.
        """
        kw_args = {
            "ms_level": self.ms_level,
            "valid_charge": self.valid_charge,
            "preprocessing_fn": self.preprocessing_fn,
        }

        if ms_data_file.suffix.lower() == ".mzml":
            return MzmlParser(ms_data_file, **kw_args)

        if ms_data_file.suffix.lower() == ".mzxml":
            return MzxmlParser(ms_data_file, **kw_args)

        if ms_data_file.suffix.lower() == ".mgf":
            return MgfParser(ms_data_file, **kw_args)

        raise ValueError("Only mzML, mzXML, and MGF files are supported.")

    def _assemble_metadata(
        self,
        parser: MzmlParser | MzxmlParser | MgfParser,
    ) -> np.ndarray:
        """Assemble the metadata.

        Parameters
        ----------
        parser : MzmlParser, MzxmlParser, MgfParser
            The parser to use.

        Returns
        -------
        numpy.ndarray of shape (n_spectra,)
            The file metadata.
        """
        meta_types = [
            ("precursor_mz", np.float32),
            ("precursor_charge", np.uint8),
            ("offset", np.uint64),
            ("scan_id", np.uint32),
        ]

        metadata = np.empty(parser.n_spectra, dtype=meta_types)
        metadata["precursor_mz"] = parser.precursor_mz
        metadata["precursor_charge"] = parser.precursor_charge
        metadata["offset"] = parser.offset
        metadata["scan_id"] = parser.scan_id
        return metadata

    def add_file(self, ms_data_file: PathLike) -> None:
        """Add a mass spectrometry data file to the index.

        Parameters
        ----------
        ms_data_file : PathLike
            The mass spectrometry data file to add. It must be in an mzML or
            MGF file format and use an ``.mzML``, ``.mzXML``, or ``.mgf`` file
            extension.
        """
        ms_data_file = Path(ms_data_file)
        if str(ms_data_file) in self._file_map:
            return

        # Invalidate current offsets:
        self._offsets = None

        # Read the file:
        parser = self._get_parser(ms_data_file)
        parser.read()

        # Create the tables:
        metadata = self._assemble_metadata(parser)

        spectrum_types = [
            ("mz_array", np.float64),
            ("intensity_array", np.float32),
        ]

        spectra = np.zeros(parser.n_peaks, dtype=spectrum_types)
        spectra["mz_array"] = parser.mz_arrays
        spectra["intensity_array"] = parser.intensity_arrays

        # Write the tables:
        with h5py.File(self.path, "a") as index:
            group_index = len(index)
            group = index.create_group(str(group_index))
            group.attrs["path"] = str(ms_data_file)
            group.attrs["n_spectra"] = parser.n_spectra
            group.attrs["n_peaks"] = parser.n_peaks
            group.attrs["id_type"] = parser.id_type

            # Update overall stats:
            index.attrs["n_spectra"] += parser.n_spectra
            index.attrs["n_peaks"] += parser.n_peaks

            # Add the datasets:
            group.create_dataset(
                "metadata",
                data=metadata,
            )

            group.create_dataset(
                "spectra",
                data=spectra,
            )

            try:
                group.create_dataset(
                    "annotations",
                    data=parser.annotations,
                    dtype=h5py.string_dtype(),
                )
            except (KeyError, AttributeError, TypeError):
                pass

            self._file_map[str(ms_data_file)] = group_index
            end_offset = self._file_offsets[-1] + parser.n_spectra
            self._file_offsets = np.append(self._file_offsets, [end_offset])

            # Update the locations:
            grp_idx = len(self._file_offsets) - 2
            for row_idx in range(parser.n_spectra):
                lin_idx = row_idx + self._file_offsets[-2]
                self._locs[lin_idx] = (grp_idx, row_idx)

    def get_spectrum(self, idx: int) -> MassSpectrum:
        """Access a mass spectrum.

        Parameters
        ----------
        idx : int
            The index of the index of the mass spectrum to look-up.

        Returns
        -------
        tuple of numpy.ndarray
            The m/z values, intensity values, precurosr m/z, precurosr charge,
            and the spectrum annotation.
        """
        group_index, row_index = self._locs[idx]
        if self._handle is None:
            raise RuntimeError("Use the context manager for access.")

        grp = self._handle[str(group_index)]
        metadata = grp["metadata"]
        spectra = grp["spectra"]
        offsets = self.offsets[str(group_index)][row_index : row_index + 2]

        start_offset = offsets[0]
        if offsets.shape[0] == 2:
            stop_offset = offsets[1]
        else:
            stop_offset = spectra.shape[0]

        spectrum = spectra[start_offset:stop_offset]
        precursor = metadata[row_index]
        return MassSpectrum(
            filename=grp.attrs["path"],
            scan_id=f"{grp.attrs['id_type']}={metadata[row_index]['scan_id']}",
            mz=np.array(spectrum["mz_array"]),
            intensity=np.array(spectrum["intensity_array"]),
            precursor_mz=precursor["precursor_mz"],
            precursor_charge=precursor["precursor_charge"],
        )

    def get_spectrum_id(self, idx: int) -> tuple[str, str]:
        """Get the identifier for a mass spectrum.

        Parameters
        ----------
        idx : int
            The index of the mass spectrum in the SpectrumIndex.

        Returns
        -------
        ms_data_file : str
            The mass spectrometry data file from which the mass spectrum was
            originally parsed.
        identifier : str
            The mass spectrum identifier, per PSI recommendations.
        """
        group_index, row_index = self._locs[idx]
        if self._handle is None:
            raise RuntimeError("Use the context manager for access.")

        grp = self._handle[str(group_index)]
        ms_data_file = grp.attrs["path"]
        identifier = grp["metadata"][row_index]["scan_id"]
        prefix = grp.attrs["id_type"]
        return ms_data_file, f"{prefix}={identifier}"

    def loader(self, *args: tuple, **kwargs: dict) -> DataLoader:
        """A PyTorch DataLoader for the mass spectra.

        Parameters
        ----------
        *args : tuple
            Arguments passed initialize a torch.utils.data.DataLoader,
            excluding ``dataset`` and ``collate_fn``.
        **kwargs : dict
            Keyword arguments passed initialize a torch.utils.data.DataLoader,
            excluding ``dataset`` and ``collate_fn``.

        Returns
        -------
        torch.utils.data.DataLoader
            A DataLoader for the mass spectra.
        """
        return DataLoader(self, *args, collate_fn=self.collate_fn, **kwargs)

    def __len__(self) -> int:
        """The number of spectra in the index."""
        return self.n_spectra

    def __del__(self) -> None:
        """Cleanup the temporary directory."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def __getitem__(self, idx: int) -> MassSpectrum:
        """Access a mass spectrum.

        Parameters
        ----------
        idx : int
            The overall index of the mass spectrum to retrieve.

        Returns
        -------
        tuple of numpy.ndarray
            The m/z values, intensity values, precursor m/z, precurosr charge,
            and the annotation (if available).
        """
        if self._handle is None:
            with self:
                return self.get_spectrum(idx)

        return self.get_spectrum(idx)

    def __enter__(self) -> SpectrumDataset:
        """Open the index file for reading."""
        if self._handle is None:
            self._handle = h5py.File(
                self.path,
                "r",
                rdcc_nbytes=int(3e8),
                rdcc_nslots=1024000,
            )
        return self

    def __exit__(self, *args: str) -> None:
        """Close the HDF5 file."""
        self._handle.close()
        self._handle = None

    @property
    def ms_files(self) -> list[str]:
        """The files currently in the index."""
        return list(self._file_map.keys())

    @property
    def path(self) -> Path:
        """The path to the underyling HDF5 index file."""
        return self._path

    @property
    def ms_level(self) -> int:
        """The MS level of tandem mass spectra in the collection."""
        return self._ms_level

    @property
    def preprocessing_fn(self) -> list[Callable]:
        """The functions for preprocessing MS data."""
        return self._preprocessing_fn

    @property
    def valid_charge(self) -> list[int]:
        """Valid precursor charges for spectra to be included."""
        return self._valid_charge

    @property
    def annotated(self) -> bool:
        """Whether or not the index contains spectrum annotations."""
        return self._annotated

    @property
    def overwrite(self) -> bool:
        """Overwrite a previous index?."""
        return self._overwrite

    @property
    def n_spectra(self) -> int:
        """The total number of mass spectra in the index."""
        if self._handle is None:
            with self:
                return self._handle.attrs["n_spectra"]

        return self._handle.attrs["n_spectra"]

    @property
    def n_peaks(self) -> int:
        """The total number of mass peaks in the index."""
        if self._handle is None:
            with self:
                return self._handle.attrs["n_peaks"]

        return self._handle.attrs["n_peaks"]

    @property
    def offsets(self) -> dict[str, np.array]:
        """The offsets denoting where each spectrum starts."""
        if self._offsets is not None:
            return self._offsets

        self._offsets = {
            k: v["metadata"]["offset"] for k, v in self._handle.items()
        }

        return self._offsets

    @staticmethod
    def collate_fn(
        batch: Iterable[MassSpectrum],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The collate function for a SpectrumDataset.

        The mass spectra must be padded so that they fit nicely as a tensor.
        However, the padded elements are ignored during the subsequent steps.

        Parameters
        ----------
        batch : tuple of tuple of torch.Tensor
            A batch of data from an SpectrumDataset.

        Returns
        -------
        spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
            The mass spectra to sequence, where ``X[:, :, 0]`` are the m/z
            values and ``X[:, :, 1]`` are their associated intensities.
        precursors : torch.Tensor of shape (batch_size, 2)
            The precursor mass and charge state.
        """
        return _collate_fn(batch)[0:2]


class AnnotatedSpectrumDataset(SpectrumDataset):
    """Store and access a collection of annotated mass spectra.

    This class parses one or more MGF file and converts it to our HDF5 index
    format. This allows us to access spectra from many different files quickly
    and without loading them all into memory.

    Parameters
    ----------
    index_path : str
        The name and path of the HDF5 file index. If the path does
        not contain the `.h5` or `.hdf5` extension, `.hdf5` will be added.
    ms_data_files : str or list of str, optional
        The MGF to include in this collection.
    ms_level : int, optional
        The level of tandem mass spectra to use.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra. ``None``,
        the default, filters for the top 200 peaks above m/z 140,
        square root transforms the intensities and scales them to unit norm.
        See the preprocessing module for details and additional options.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    overwrite : bool
        Overwrite previously indexed files? If ``False`` and new files are
        provided, they will be appended to the collection.

    Attributes
    ----------
    ms_files : list of str
    path : Path
    ms_level : int
    valid_charge : Optional[Iterable[int]]
    overwrite : bool
    n_spectra : int
    n_peaks : int
    """

    _annotated = True

    def __init__(
        self,
        tokenizer: PeptideTokenizer,
        ms_data_files: PathLike | Iterable[PathLike] = None,
        ms_level: int = 2,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        index_path: PathLike | None = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize an AnnotatedSpectrumIndex."""
        self.tokenizer = tokenizer
        super().__init__(
            ms_data_files=ms_data_files,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            index_path=index_path,
            overwrite=overwrite,
        )

    def _get_parser(self, ms_data_file: str) -> MgfParser:
        """Get the parser for the MS data file."""
        if ms_data_file.suffix.lower() == ".mgf":
            return MgfParser(
                ms_data_file,
                ms_level=self.ms_level,
                annotations=True,
                preprocessing_fn=self.preprocessing_fn,
            )

        raise ValueError("Only MGF files are currently supported.")

    def get_spectrum(self, idx: int) -> MassSpectrum:
        """Access a mass spectrum.

        Parameters
        ----------
        idx : int
            The index of the mass spectrum in the AnnotatedSpectrumIndex.

        Returns
        -------
        MassSpectrum
            The mass spectrum, labeled with its annotation.
        """
        spectrum = super().get_spectrum(idx)
        group_index, row_index = self._locs[idx]
        grp = self._handle[str(group_index)]
        annotations = grp["annotations"]
        spectrum.label = annotations[row_index].decode()
        return spectrum

    @property
    def annotations(self) -> np.ndarray[str]:
        """Retrieve all of the annotations in the index."""
        annotations = []
        for grp in self._handle.values():
            try:
                annotations.append(grp["annotations"])
            except KeyError:
                pass

        return np.concatenate(annotations)

    def collate_fn(
        self,
        batch: Iterable[MassSpectrum],
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray[str]]:
        """The collate function for a AnnotatedSpectrumDataset.

        The mass spectra must be padded so that they fit nicely as a tensor.
        However, the padded elements are ignored during the subsequent steps.

        Parameters
        ----------
        batch : tuple of tuple of torch.Tensor
            A batch of data from an AnnotatedSpectrumDataset.

        Returns
        -------
        spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
            The mass spectra to sequence, where ``X[:, :, 0]`` are the m/z
            values and ``X[:, :, 1]`` are their associated intensities.
        precursors : torch.Tensor of shape (batch_size, 2)
            The precursor mass and charge state.
        annotations : np.ndarray[str]
            The spectrum annotations.
        """
        spectra, precursors, annotations = _collate_fn(batch)
        tokens = self.tokenizer.tokenize(annotations)
        return spectra, precursors, tokens


def _collate_fn(
    batch: Iterable[MassSpectrum],
) -> tuple[torch.Tensor, torch.Tensor, list[str | None]]:
    """The collate function for a SpectrumDataset.

    The mass spectra must be padded so that they fit nicely as a tensor.
    However, the padded elements are ignored during the subsequent steps.

    Parameters
    ----------
    batch : tuple of tuple of torch.Tensor
        A batch of data from an SpectrumDataset.

    Returns
    -------
    spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
        The mass spectra to sequence, where ``X[:, :, 0]`` are the m/z
        values and ``X[:, :, 1]`` are their associated intensities.
    precursors : torch.Tensor of shape (batch_size, 2)
        The precursor mass and charge state.
    annotations : np.ndarray of shape (batch_size,)
        The annotations, if any exist.
    """
    spectra = []
    masses = []
    charges = []
    annotations = []
    for spec in batch:
        spectra.append(spec.to_tensor())
        masses.append(spec.precursor_mass)
        charges.append(spec.precursor_charge)
        annotations.append(spec.label)

    precursors = torch.vstack([torch.tensor(masses), torch.tensor(charges)])
    spectra = torch.nn.utils.rnn.pad_sequence(
        spectra,
        batch_first=True,
    )
    return spectra, precursors.T.float(), annotations


def _hash_obj(obj: Any) -> str:  # noqa: ANN401
    """SHA1 hash for a picklable object."""
    out = hashlib.sha1()
    out.update(dill.dumps(obj))
    return out.hexdigest()
