"""Mass spectrometry data parsers"""
import logging
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
from pyteomics.mzml import MzML
from pyteomics.mzxml import MzXML
from pyteomics.mgf import MGF


LOGGER = logging.getLogger(__name__)


class BaseParser(ABC):
    """A base parser class to inherit from.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    id_type : str, optional
        The Hupo-PSI prefix for the spectrum identifier.
    """

    def __init__(
        self,
        ms_data_file,
        ms_level,
        valid_charge=None,
        id_type="scan",
    ):
        """Initialize the BaseParser"""
        self.path = Path(ms_data_file)
        self.ms_level = ms_level
        self.valid_charge = None if valid_charge is None else set(valid_charge)
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
        """Read the ms data file.

        Returns
        -------
        Self
        """
        n_skipped = 0
        with self.open() as spectra:
            for spectrum in tqdm(spectra, desc=str(self.path), unit="spectra"):
                try:
                    self.parse_spectrum(spectrum)
                except (IndexError, KeyError, ValueError):
                    n_skipped += 1

        if n_skipped:
            LOGGER.warning(
                "Skipped %d spectra with invalid precursor info", n_skipped
            )

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
        return self

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
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    """

    def __init__(self, ms_data_file, ms_level=2, valid_charge=None):
        """Initialize the MzmlParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
        )

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
            precursor = spectrum["precursorList"]["precursor"][0]
            precursor_ion = precursor["selectedIonList"]["selectedIon"][0]
            precursor_mz = float(precursor_ion["selected ion m/z"])
            if "charge state" in precursor_ion:
                precursor_charge = int(precursor_ion["charge state"])
            elif "possible charge state" in precursor_ion:
                precursor_charge = int(precursor_ion["possible charge state"])
            else:
                precursor_charge = 0
        else:
            precursor_mz, precursor_charge = None, 0

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(spectrum["m/z array"])
            self.intensity_arrays.append(spectrum["intensity array"])
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(_parse_scan_id(spectrum["id"]))
        else:
            raise ValueError("Invalid precursor charge")


class MzxmlParser(BaseParser):
    """Parse mass spectra from an mzXML file.

    Parameters
    ----------
    ms_data_file : str or Path
        The mzXML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    """

    def __init__(self, ms_data_file, ms_level=2, valid_charge=None):
        """Initialize the MzxmlParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
        )

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
            precursor = spectrum["precursorMz"][0]
            precursor_mz = float(precursor["precursorMz"])
            precursor_charge = int(precursor.get("precursorCharge", 0))
        else:
            precursor_mz, precursor_charge = None, 0

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(spectrum["m/z array"])
            self.intensity_arrays.append(spectrum["intensity array"])
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(_parse_scan_id(spectrum["id"]))
        else:
            raise ValueError("Invalid precursor charge")


class MgfParser(BaseParser):
    """Parse mass spectra from an MGF file.

    Parameters
    ----------
    ms_data_file : str or Path
        The MGF file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    annotations : bool
        Include peptide annotations.
    """

    def __init__(
        self,
        ms_data_file,
        ms_level=2,
        valid_charge=None,
        annotations=False,
    ):
        """Initialize the MgfParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
            id_type="index",
        )
        self.annotations = [] if annotations else None
        self._counter = -1

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
        self._counter += 1

        if self.ms_level > 1:
            precursor_mz = float(spectrum["params"]["pepmass"][0])
            precursor_charge = int(spectrum["params"].get("charge", [0])[0])
        else:
            precursor_mz, precursor_charge = None, 0

        if self.annotations is not None:
            self.annotations.append(spectrum["params"].get("seq"))

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(spectrum["m/z array"])
            self.intensity_arrays.append(spectrum["intensity array"])
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(self._counter)
        else:
            raise ValueError("Invalid precursor charge")


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

    raise ValueError("Failed to parse scan number")
