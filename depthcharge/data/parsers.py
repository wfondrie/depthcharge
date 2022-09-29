"""Mass spectrometry data parsers"""
import functools
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
from pyteomics.mzml import MzML
from pyteomics.mzxml import MzXML
from pyteomics.mgf import MGF


class BaseParser(ABC):
    """A base parser class to inherit from.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    id_type : str, optional
        The Hupo-PSI prefix for the spectrum identifier.
    """

    def __init__(self, ms_data_file, ms_level, id_type="scan"):
        """Initialize the BaseParser"""
        self.path = Path(ms_data_file)
        self.ms_level = ms_level

        self.id_type = id_type
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
            pbar = functools.partial(tqdm, desc=str(self.path), unit="spectra")
            for spectrum_index, spectrum in enumerate(pbar(spectra)):
                self.parse_spectrum(spectrum)

        self.precursor_mz = np.array(self.precursor_mz, dtype=np.float64)
        self.precursor_charge = np.array(
            self.precursor_charge,
            dtype=np.uint8,
        )

        self.scan_id = np.array(self.scan_id)

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
            self.precursor_charge.append(0)

        self.mz_arrays.append(spectrum["m/z array"])
        self.intensity_arrays.append(spectrum["intensity array"])
        self.scan_id.append(_parse_scan_id(spectrum["id"]))


class MzxmlParser(BaseParser):
    """Parse mass spectra from an mzXML file.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzXML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    """

    def __init__(self, ms_data_file, ms_level=2):
        """Initialize the MzxmlParser."""
        super().__init__(ms_data_file, ms_level=ms_level)

    def open(self):
        """Open the mzXML file for reading"""
        return MzXML(str(self.path))

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in mzXML format.
        """
        if spectrum["msLevel"] != self.ms_level:
            return

        if self.ms_level > 1:
            data = spectrum["precursorMz"][0]
            self.precursor_mz.append(data["precursorMz"])
            self.precursor_charge.append(data.get("precursorCharge", 0))
        else:
            self.precursor_mz.append(None)
            self.precursor_charge.append(0)

        self.mz_arrays.append(spectrum["m/z array"])
        self.intensity_arrays.append(spectrum["intensity array"])
        self.scan_id.append(_parse_scan_id(spectrum["id"]))


class MgfParser(BaseParser):
    """Parse mass spectra from an MGF file.

    Parameters
    ----------
    ms_data_file : str or Path
        The MGF file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    annotations : bool
        Include peptide annotations.
    """

    def __init__(self, ms_data_file, ms_level=2, annotations=False):
        """Initialize the MgfParser."""
        super().__init__(ms_data_file, ms_level=ms_level, id_type="index")
        self.annotations = [] if annotations else None
        self._counter = 0

    def open(self):
        """Open the MGF file for reading"""
        return MGF(str(self.path))

    def parse_spectrum(self, spectrum):
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in MGF format.
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
        self.scan_id.append(self._counter)
        self._counter += 1


def _parse_scan_id(scan_str):
    """Remove the string prefix from the scan ID.

    Adapted from:
    https://github.com/bittremieux/GLEAMS/blob/
    8831ad6b7a5fc391f8d3b79dec976b51a2279306/gleams/
    ms_io/mzml_io.py#L82-L85

    Parameters
    ----------
    scan_str : str
        The scan ID string.

    Returns
    -------
    int
        The scan ID number.
    """
    try:
        return int(scan_str)
    except ValueError:
        try:
            return int(scan_str[scan_str.find("scan=") + len("scan=") :])
        except ValueError:
            pass

    raise ValueError(f"Failed to parse scan number")
