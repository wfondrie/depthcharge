"""Mass spectrometry data parsers"""
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
from pyteomics.mzml import MzML
from pyteomics.mgf import MGF


class BaseParser(ABC):
    """A base parser class to inherit from."""

    def __init__(self, ms_data_file, ms_level):
        """Initialize the BaseParser"""
        self.path = Path(ms_data_file)
        self.ms_level = ms_level

        self.offset = None
        self.precursor_mz = []
        self.precursor_charge = []
        self.scan_id = []
        self.mz_arrays = []
        self.intensity_arrays = []

    @abstractmethod
    def open(self):
        """Open the file as an iterable"""
        pass

    @abstractmethod
    def parse_spectrum(self, spectrum):
        """Parse a single spectrum

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in a given format.
        """
        pass

    def read(self):
        """Read the ms data file"""
        with self.open() as spectra:
            for spectrum_index, spectrum in enumerate(tqdm(spectra, desc=str(self.path), unit="spectra")):
                self.parse_spectrum(spectrum, spectrum_index)

        self.precursor_mz = np.array(self.precursor_mz, dtype=np.float64)
        self.precursor_charge = np.array(
            self.precursor_charge,
            dtype=np.uint8,
        )

        self.scan_id = np.array(
            self.scan_id,
            dtype=np.uint64,
        )

        # Build the index
        sizes = np.array([0] + [s.shape[0] for s in self.mz_arrays])
        self.offset = sizes[:-1].cumsum()
        self.mz_arrays = np.concatenate(self.mz_arrays).astype(np.float64)
        self.intensity_arrays = np.concatenate(self.intensity_arrays).astype(
            np.float32
        )

    @property
    def n_spectra(self):
        """The number of spectra"""
        return self.offset.shape[0]

    @property
    def n_peaks(self):
        """The number of peaks in the file."""
        return self.mz_arrays.shape[0]


class MzmlParser(BaseParser):
    """Parse mass spectra from an mzML file.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    """

    def __init__(self, ms_data_file, ms_level=2):
        """Initialize the MzmlParser."""
        super().__init__(ms_data_file, ms_level=ms_level)

    def open(self):
        """Open the mzML file for reading"""
        return MzML(str(self.path))

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in mzML format.
        """
        if spectrum["ms level"] != self.ms_level:
            return

        if self.ms_level > 1:
            data = (
                spectrum.get("precursorList")
                .get("precursor")[0]
                .get("selectedIonList")
                .get("selectedIon")[0]
            )
            self.precursor_mz.append(data["selected ion m/z"])
            self.precursor_charge.append(data.get("charge state", 0))
        else:
            self.precursor_mz.append(None)
            self.precursor_charge.append(None)

        self.mz_arrays.append(spectrum["m/z array"])
        self.intensity_arrays.append(spectrum["intensity array"])
        
        if "scans" in spectrum["params"].keys():
            self.scan_id.append(spectrum["params"]["scans"])
        else:
            self.scan_id.append(spectrum_index)
        
        
class MgfParser(BaseParser):
    """Parse mass spectra from an mzML file.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    annotations : bool
        Include peptide annotations.
    """

    def __init__(self, ms_data_file, ms_level=2, annotations=False):
        """Initialize the MzmlParser."""
        super().__init__(ms_data_file, ms_level=ms_level)
        self.annotations = [] if annotations else None

    def open(self):
        """Open the mzML file for reading"""
        return MGF(str(self.path))

    def parse_spectrum(self, spectrum, spectrum_index):
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in mzML format.
        """
        if self.ms_level > 1:
            self.precursor_mz.append(spectrum["params"]["pepmass"][0])
            self.precursor_charge.append(spectrum["params"]["charge"][0])
        else:
            self.precursor_mz.append(None)
            self.precursor_charge.append(None)

        if self.annotations is not None:
            self.annotations.append(spectrum["params"]["seq"])

        self.mz_arrays.append(spectrum["m/z array"])
        self.intensity_arrays.append(spectrum["intensity array"])
        
        if "scans" in spectrum["params"].keys():
            self.scan_id.append(spectrum["params"]["scans"])
        else:
            self.scan_id.append(spectrum_index)
        
        