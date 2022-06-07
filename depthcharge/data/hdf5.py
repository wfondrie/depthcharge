"""Parse mass spectra into an HDF5 file format."""
import logging
from pathlib import Path

import h5py
import torch
import numpy as np

from .. import utils
from .parsers import MzmlParser, MgfParser

LOGGER = logging.getLogger(__name__)


class SpectrumIndex:
    """Store and access a collection of mass spectra.

    This class parses one or more mzML file, converts it to an HDF5 file
    format. This allows depthcharge to access spectra from many different
    files quickly and without loading them all into memory.

    Parameters
    ----------
    index_path : str
        The name and path of the HDF5 file index. If the path does
        not contain the `.h5` or `.hdf5` extension, `.hdf5` will be added.
    ms_data_files : str or list of str, optional
        The mzML to include in this collection.
    ms_level : int, optional
        The level of tandem mass spectra to use.
    metadata : dict or list of dict, optional
        Additional metadata to store with the files.
    overwite : bool, optional
        Overwrite previously indexed files? If ``False`` and new files are
        provided, they will be appended to the collection.

    Attributes
    ----------
    ms_files : list of str
    path : Path
    ms_level : int
    overwrite : bool
    n_spectra : int
    n_peaks : int
    """

    def __init__(
        self,
        index_path,
        ms_data_files=None,
        ms_level=2,
        overwrite=False,
    ):
        """Initialize a SpectrumIndex"""
        index_path = Path(index_path)
        if index_path.suffix not in [".h5", ".hdf5"]:
            index_path = Path(str(index_path) + ".hdf5")

        # Set attributes and check parameters:
        self._path = index_path
        self._ms_level = utils.check_positive_int(ms_level, "ms_level")
        self._overwrite = bool(overwrite)
        self._handle = None
        self._file_offsets = np.array([0])
        self._file_map = {}

        # Create the file if it doesn't exist.
        if not self.path.exists() or self.overwrite:
            with h5py.File(self.path, "w") as index:
                index.attrs["ms_level"] = self.ms_level
                index.attrs["n_spectra"] = 0
                index.attrs["n_peaks"] = 0
                index.attrs["annotated"] = False

        # Else, verify that the previous index uses the same ms_level.
        else:
            with self:
                try:
                    assert self._handle.attrs["ms_level"] == self.ms_level
                    assert self._handle.attrs["annotated"] == self.annotated
                except (KeyError, AssertionError):
                    raise ValueError(
                        f"'{self.path}' already exists, but was created with "
                        "incompatible parameters. Use 'overwrite=True' to "
                        "overwrite it."
                    )

                self._reindex()

        # Now parse spectra.
        if ms_data_files is not None:
            ms_data_files = utils.listify(ms_data_files)
            LOGGER.info("Reading %i files...", len(ms_data_files))
            for ms_file in ms_data_files:
                self.add_file(ms_file)

    def _reindex(self):
        """Update the file mappings and offsets."""
        offsets = []
        for idx in range(len(self._handle)):
            grp = self._handle[str(idx)]
            offsets.append(grp.attrs["n_spectra"])
            self._file_map[grp.attrs["path"]] = idx

        self._file_offsets = np.cumsum([0] + offsets)

    def _get_parser(self, ms_data_file):
        """Get the parser for the MS data file.

        Parameters
        ----------
        ms_data_file : Path
            The mass spectrometry data file to be parsed.

        Returns
        -------
        MzmlParser or MgfParser
            The appropriate parser for the file.
        """
        if ms_data_file.suffix.lower() == ".mzml":
            return MzmlParser(ms_data_file, ms_level=self.ms_level)

        if ms_data_file.suffix.lower() == ".mgf":
            return MgfParser(ms_data_file, ms_level=self.ms_level)

        raise ValueError(f"Only mzML and MGF files are supported.")

    def _assemble_metadata(self, parser):
        """Assemble the metadata.

        Parameters
        ----------
        parser : MzmlParser or MgfParser
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
            ("scan_id", np.uint64),
        ]

        metadata = np.empty(parser.n_spectra, dtype=meta_types)
        metadata["precursor_mz"] = parser.precursor_mz
        metadata["precursor_charge"] = parser.precursor_charge
        metadata["offset"] = parser.offset
        metadata["scan_id"] = parser.scan_id
        return metadata

    def add_file(self, ms_data_file):
        """Add a mass spectrometry data file to the index.

        Parameters
        ----------
        ms_data_file : str or Path
            The mass spectrometry data file to add. It must be in an mzML or
            MGF file format and use an ``.mzML`` or ``.mgf`` file extension.
        """
        ms_data_file = Path(ms_data_file)
        if str(ms_data_file) in self._file_map:
            return

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
            except KeyError:
                pass

            self._file_map[str(ms_data_file)] = group_index
            end_offset = self._file_offsets[-1] + parser.n_spectra
            self._file_offsets = np.append(self._file_offsets, [end_offset])

    def get_spectrum(self, file_index, spectrum_index):
        """Access a mass spectrum.

        Parameters
        ----------
        file_index : int
            The group index of the MS data file.
        spectrum_index : int
            The index of the mass spectrum within the file.

        Returns
        -------
        tuple of numpy.ndarray
            The m/z values, intensity values, precurosr m/z, precurosr charge,
            and the spectrum annotation.
        """
        if self._handle is None:
            raise RuntimeError("Use the context manager for access.")

        grp = self._handle[str(file_index)]
        metadata = grp["metadata"]
        spectra = grp["spectra"]
        offsets = metadata["offset"][spectrum_index : spectrum_index + 2]

        start_offset = offsets[0]
        if offsets.shape[0] == 2:
            stop_offset = offsets[1]
        else:
            stop_offset = spectra.shape[0]

        spectrum = spectra[start_offset:stop_offset]
        precursor = metadata[spectrum_index]
        out = (
            spectrum["mz_array"],
            spectrum["intensity_array"],
            precursor["precursor_mz"],
            precursor["precursor_charge"],
            precursor["scan_id"],
        )

        return out

    def __getitem__(self, idx):
        """Access a mass spectrum.

        Parameters
        ----------
        idx : int
            The overall index of the mass spectrum to retrieve.

        Returns
        -------
        tuple of numpy.ndarray
            The m/z values, intensity values, precurosr m/z, precurosr charge,
            and the annotation (if available).
        """
        file_index = np.searchsorted(self._file_offsets[1:], idx, side="right")
        spectrum_index = idx - self._file_offsets[file_index]

        if self._handle is None:
            with self:
                return self.get_spectrum(file_index, spectrum_index)

        return self.get_spectrum(file_index, spectrum_index)

    def __enter__(self):
        """Open the index file for reading."""
        self._handle = h5py.File(
            self.path,
            "r",
            rdcc_nbytes=int(3e8),
            rdcc_nslots=1024000,
        )
        return self

    def __exit__(self, *args):
        """Close the HDF5 file."""
        self._handle.close()
        self._handle = None

    @property
    def ms_files(self):
        """The files currently in the index."""
        return list(self._file_map.keys())

    @property
    def path(self):
        """The path to the underyling HDF5 index file."""
        return self._path

    @property
    def ms_level(self):
        """The MS level of tandem mass spectra in the collection."""
        return self._ms_level

    @property
    def overwrite(self):
        """Overwrite a previous index?"""
        return self._overwrite

    @property
    def n_spectra(self):
        """The total number of mass spectra in the index."""
        if self._handle is None:
            with self:
                return self._handle.attrs["n_spectra"]

        return self._handle.attrs["n_spectra"]

    @property
    def n_peaks(self):
        """The total number of mass peaks in the index."""
        if self._handle is None:
            with self:
                return self._handle.attrs["n_peaks"]

        return self._handle.attrs["n_peaks"]


class AnnotatedSpectrumIndex(SpectrumIndex):
    """Store and access a collection of annotated mass spectra.

    This class parses one or more mzML file, converts it to our HDF5 index
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
    metadata : dict or list of dict, optional
        Additional metadata to store with the files.
    overwite : bool
        Overwrite previously indexed files? If ``False`` and new files are
        provided, they will be appended to the collection.

    Attributes
    ----------
    ms_files : list of str
    path : Path
    ms_level : int
    overwrite : bool
    n_spectra : int
    n_peaks : int
    """

    def __init__(
        self,
        index_path,
        ms_data_files=None,
        ms_level=2,
        overwrite=False,
    ):
        """Initialize an AnnotatedSpectrumIndex"""
        super().__init__(
            index_path=index_path,
            ms_data_files=ms_data_files,
            ms_level=ms_level,
            overwrite=overwrite,
        )

    def _get_parser(self, ms_data_file):
        """Get the parser for the MS data file"""
        if ms_data_file.suffix.lower() == ".mgf":
            return MgfParser(
                ms_data_file,
                ms_level=self.ms_level,
                annotations=True,
            )

        raise ValueError("Only MGF files are supported.")

    def get_spectrum(self, file_index, spectrum_index):
        """Access a mass spectrum.

        Parameters
        ----------
        file_index : int
            The group index of the MS data file.
        spectrum_index : int
            The index of the mass spectrum within the file.

        Returns
        -------
        tuple of numpy.ndarray
            The m/z values, intensity values, precurosr m/z, precurosr charge,
            and the spectrum annotation.
        """
        spec_info = super().get_spectrum(file_index, spectrum_index)
        grp = self._handle[str(file_index)]
        annotations = grp["annotations"]
        spec_ann = annotations[spectrum_index].decode()
        return (*spec_info, spec_ann)
