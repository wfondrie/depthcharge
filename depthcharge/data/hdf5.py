"""Parse mass spectra into an HDF5 file format."""
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
from pyteomics.mzml import MzML
from pyteomics.mgf import MGF

from .. import utils
from .parsers import MzmlParser, MgfParser

LOGGER = logging.getLogger(__name__)


class SpectrumIndex:
    """Store and access a collection of mass spectra.

    This class parses one or more mzML file, converts it to our HDF5 index
    format. It creates a temporary HDF5 file with external mappings to all of
    the mzML indices that comprise it, allowing us to quickly access
    individual spectra.

    Parameters
    ----------
    index_path : str
        The name and path of the new HDF5 file index. If the path does
        not contain the `.h5` or `.hdf5` extension, `.hdf5` will be added.
    ms_data_files : str or list of str, optional
        The mzML to include in this collection.
    ms_level : int, optional
        The level of tandem mass spectra to use.
    metadata : dict or list of dict, optional
        Additional metadata to store with the files.
    overwite : bool
        Overwrite previously indexed files? If ``False`` and new files are
        provided, they will be appended to the collection.

    Attributes
    ----------
    path : Path
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
        ms_data_files = utils.listify(ms_data_files)

        # Create the file if it doesn't exist.
        if not self.path.exists() or self.overwrite:
            with h5py.File(self.path, "w") as index:
                index.attrs["ms_level"] = self.ms_level

        # Else, verify that the previous index uses the same ms_level.
        else:
            with h5py.File(self.path, "a") as index:
                try:
                    assert index.attrs["ms_level"] == self.ms_level
                except (KeyError, AssertionError):
                    ValueError(
                        f"'{self.path}' already exists, but was created with "
                        "incompatible parameters. Use 'overwrite=True' to "
                        "overwrite it."
                    )

        # Now parse spectra.
        LOGGER.info("Reading %i files...", len(ms_data_files))
        for ms_file in ms_data_files:
            self.add_file(ms_file)

    def add_file(self, ms_data_file):
        """Add a MS data file to the index"""
        ms_data_file = Path(ms_data_file)
        if ms_data_file.suffix.lower() == ".mzml":
            parser = MzmlParser(ms_data_file, ms_level=2)
        elif ms_data_file.suffix.lower() == ".mgf":
            parser = MgfParser(ms_data_file, ms_level=2)

        # Read the file:
        parser.read()

        # Create the tables:
        meta_types = [
            ("precursor_mz", np.float32),
            ("precursor_charge", np.uint8),
            ("offset", np.uint64),
        ]

        metadata = np.zeros(3, dtype=meta_types)
        metadata["precursor_mz"] = parser.precursor_mz
        metadata["precursor_charge"] = parser.precursor_charge
        metadata["offset"] = parser.offset

        spectrum_types = [
            ("mz_array", np.float64),
            ("intensity_array", np.float32),
        ]

        spectra = np.zeros(2, dtype=spectrum_types)
        spectra["mz_array"] = parser.mz_arrays
        spectra["intensity_array"] = parser.intensity_arrays

        # Write the tables:
        with h5py.File(self.path, "a") as index:
            group = index.create_group(str(len(index)))
            group.attrs["path"] = str(ms_data_file)
            group.attrs["n_spectra"] = parser.n_spectra
            group.attrs["n_peaks"] = parser.n_peaks
            group.create_dataset("metadata", data=metadata, chunks=True)
            group.create_dataset("spectra", data=spectra, chunks=True)

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
