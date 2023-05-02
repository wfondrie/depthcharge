"""Mass spectrometry data parsers"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import spectrum_utils.spectrum as sus
from pyteomics.mgf import MGF
from pyteomics.mzml import MzML
from pyteomics.mzxml import MzXML
from tqdm.auto import tqdm


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
    min_mz : Optional[float]
        Remove spectrum peaks below the given minimum m/z.
    max_mz : Optional[float]
        Remove spectrum peaks above the given maximum m/z.
    min_intensity : Optional[float]
        Remove spectrum peaks below the given relative minimum intensity.
    max_n_peaks : Optional[int]
        Only keep the specified number of most intense peaks in each spectrum.
    remove_precursor_tol : Optional[float]
        Remove spectrum peaks in the given m/z tolerance around the precursor
        m/z.
    id_type : str, optional
        The Hupo-PSI prefix for the spectrum identifier.
    """

    def __init__(
        self,
        ms_data_file,
        ms_level,
        valid_charge=None,
        min_mz=None,
        max_mz=None,
        min_intensity=None,
        max_n_peaks=None,
        remove_precursor_tol=None,
        id_type="scan",
    ):
        """Initialize the BaseParser"""
        self.path = Path(ms_data_file)
        self.ms_level = ms_level
        self.valid_charge = None if valid_charge is None else set(valid_charge)
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = 0.0 if min_intensity is None else min_intensity
        self.max_n_peaks = max_n_peaks
        self.remove_precursor_tol = remove_precursor_tol
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
    def parse_spectrum(self, spectrum) -> sus.MsmsSpectrum:
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in a given format.

        Returns
        -------
        sus.MsmsSpectrum
            The parsed spectrum as an MsmsSpectrum object.

        Raises
        ------
        ValueError
            If the spectrum properties are invalid (unsupported MS level or
            precursor charge).
        """
        pass

    def read(self):
        """Read the ms data file."""
        n_skipped = 0
        with self.open() as spectra:
            for spectrum in tqdm(spectra, desc=str(self.path), unit="spectra"):
                try:
                    spectrum = self.parse_spectrum(spectrum)
                    mz_array, int_array = self._process_peaks(spectrum)

                    self.scan_id.append(_parse_scan_id(spectrum.identifier))
                    self.precursor_mz.append(spectrum.precursor_mz)
                    self.precursor_charge.append(spectrum.precursor_charge)
                    self.mz_arrays.append(mz_array)
                    self.intensity_arrays.append(int_array)
                except (IndexError, KeyError, ValueError):
                    n_skipped += 1

        if n_skipped:
            LOGGER.warning(
                "Skipped %d spectra with invalid properties", n_skipped
            )

        self.precursor_mz = np.array(self.precursor_mz, dtype=np.float64)
        self.precursor_charge = np.array(self.precursor_charge, dtype=np.uint8)
        self.scan_id = np.array(self.scan_id)

        # Build the index
        sizes = np.array([0] + [s.shape[0] for s in self.mz_arrays])
        self.offset = sizes[:-1].cumsum()
        self.mz_arrays = np.concatenate(self.mz_arrays).astype(np.float64)
        self.intensity_arrays = np.concatenate(self.intensity_arrays).astype(
            np.float32
        )

    def _process_peaks(
        self, spectrum: sus.MsmsSpectrum
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the spectrum by removing noise peaks and scaling the peak
        intensities.

        Parameters
        ----------
        spectrum: sus.MsmsSpectrum
            The spectrum to be preprocessed.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
        mz_array:np.ndarray
            The spectrum peak m/z values after preprocessing.
        int_array:np.ndarray
            The spectrum peak intensity values after preprocessing.

        Raises
        ------
        ValueError
            If no peaks are remaining after preprocessing.
        TODO: Do we want to configure a minimum number of peaks for a spectrum to be valid?
        """
        spectrum.set_mz_range(self.min_mz, self.max_mz)
        if len(spectrum.mz) == 0:
            raise ValueError
        if self.remove_precursor_tol is not None:
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
            if len(spectrum.mz) == 0:
                raise ValueError
        spectrum.filter_intensity(self.min_intensity, self.max_n_peaks)
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.scale_intensity("root", 1)
        intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)
        return spectrum.mz, intensities

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
    min_mz : Optional[float]
        Remove spectrum peaks below the given minimum m/z.
    max_mz : Optional[float]
        Remove spectrum peaks above the given maximum m/z.
    min_intensity : Optional[float]
        Remove spectrum peaks below the given relative minimum intensity.
    max_n_peaks : Optional[int]
        Only keep the specified number of most intense peaks in each spectrum.
    remove_precursor_tol : Optional[float]
        Remove spectrum peaks in the given m/z tolerance around the precursor
        m/z.
    """

    def __init__(
        self,
        ms_data_file,
        ms_level=2,
        valid_charge=None,
        min_mz=None,
        max_mz=None,
        min_intensity=None,
        max_n_peaks=None,
        remove_precursor_tol=None,
    ):
        """Initialize the MzmlParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            max_n_peaks=max_n_peaks,
            remove_precursor_tol=remove_precursor_tol,
        )

    def open(self):
        """Open the mzML file for reading"""
        return MzML(str(self.path))

    def parse_spectrum(self, spectrum) -> sus.MsmsSpectrum:
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in mzML format.

        Returns
        -------
        sus.MsmsSpectrum
            The parsed spectrum as an MsmsSpectrum object.

        Raises
        ------
        ValueError
            If the spectrum properties are invalid (unsupported MS level or
            precursor charge).
        """
        if spectrum["ms level"] != self.ms_level:
            raise ValueError("Unsupported MS level")

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

        if (
                self.valid_charge is not None
                and precursor_charge not in self.valid_charge
        ):
            raise ValueError("Unsupported precursor charge")

        return sus.MsmsSpectrum(
            spectrum["id"],
            precursor_mz,
            precursor_charge,
            spectrum["m/z array"],
            spectrum["intensity array"]
        )


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
    min_mz : Optional[float]
        Remove spectrum peaks below the given minimum m/z.
    max_mz : Optional[float]
        Remove spectrum peaks above the given maximum m/z.
    min_intensity : Optional[float]
        Remove spectrum peaks below the given relative minimum intensity.
    max_n_peaks : Optional[int]
        Only keep the specified number of most intense peaks in each spectrum.
    remove_precursor_tol : Optional[float]
        Remove spectrum peaks in the given m/z tolerance around the precursor
        m/z.
    """

    def __init__(
        self,
        ms_data_file,
        ms_level=2,
        valid_charge=None,
        min_mz=None,
        max_mz=None,
        min_intensity=None,
        max_n_peaks=None,
        remove_precursor_tol=None,
    ):
        """Initialize the MzxmlParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            max_n_peaks=max_n_peaks,
            remove_precursor_tol=remove_precursor_tol,
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

        Returns
        -------
        sus.MsmsSpectrum
            The parsed spectrum as an MsmsSpectrum object.

        Raises
        ------
        ValueError
            If the spectrum properties are invalid (unsupported MS level or
            precursor charge).
        """
        if spectrum["msLevel"] != self.ms_level:
            raise ValueError("Unsupported MS level")

        if self.ms_level > 1:
            precursor = spectrum["precursorMz"][0]
            precursor_mz = float(precursor["precursorMz"])
            precursor_charge = int(precursor.get("precursorCharge", 0))
        else:
            precursor_mz, precursor_charge = None, 0

        if (
                self.valid_charge is not None
                and precursor_charge not in self.valid_charge
        ):
            raise ValueError("Unsupported precursor charge")

        return sus.MsmsSpectrum(
            spectrum["id"],
            precursor_mz,
            precursor_charge,
            spectrum["m/z array"],
            spectrum["intensity array"]
        )


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
    min_mz : Optional[float]
        Remove spectrum peaks below the given minimum m/z.
    max_mz : Optional[float]
        Remove spectrum peaks above the given maximum m/z.
    min_intensity : Optional[float]
        Remove spectrum peaks below the given relative minimum intensity.
    max_n_peaks : Optional[int]
        Only keep the specified number of most intense peaks in each spectrum.
    remove_precursor_tol : Optional[float]
        Remove spectrum peaks in the given m/z tolerance around the precursor
        m/z.
    annotations : bool
        Include peptide annotations.
    """

    def __init__(
        self,
        ms_data_file,
        ms_level=2,
        valid_charge=None,
        min_mz=None,
        max_mz=None,
        min_intensity=None,
        max_n_peaks=None,
        remove_precursor_tol=None,
        annotations=False,
    ):
        """Initialize the MgfParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            max_n_peaks=max_n_peaks,
            remove_precursor_tol=remove_precursor_tol,
            id_type="index",
        )
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
        spectrum_id = str(self._counter)
        self._counter += 1

        if self.ms_level > 1:
            precursor_mz = float(spectrum["params"]["pepmass"][0])
            precursor_charge = int(spectrum["params"].get("charge", [0])[0])
        else:
            precursor_mz, precursor_charge = None, 0

        if (
                self.valid_charge is not None
                and precursor_charge not in self.valid_charge
        ):
            raise ValueError("Unsupported precursor charge")

        if self.annotations is not None:
            self.annotations.append(spectrum["params"].get("seq"))

        return sus.MsmsSpectrum(
            spectrum_id,
            precursor_mz,
            precursor_charge,
            spectrum["m/z array"],
            spectrum["intensity array"]
        )


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
