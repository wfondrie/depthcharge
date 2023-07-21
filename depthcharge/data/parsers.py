"""Mass spectrometry data parsers."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path

import numpy as np
from pyteomics.mgf import MGF
from pyteomics.mzml import MzML
from pyteomics.mzxml import MzXML
from tqdm.auto import tqdm

from .. import utils
from ..primitives import MassSpectrum

LOGGER = logging.getLogger(__name__)


class BaseParser(ABC):
    """A base parser class to inherit from.

    Parameters
    ----------
    ms_data_file : PathLike
        The peak file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    id_type : str, optional
        The Hupo-PSI prefix for the spectrum identifier.
    """

    def __init__(
        self,
        ms_data_file: PathLike,
        ms_level: int,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        id_type: str = "scan",
    ) -> None:
        """Initialize the BaseParser."""
        self.path = Path(ms_data_file)
        self.ms_level = ms_level
        if preprocessing_fn is None:
            self.preprocessing_fn = []
        else:
            self.preprocessing_fn = utils.listify(preprocessing_fn)

        self.valid_charge = None if valid_charge is None else set(valid_charge)
        self.id_type = id_type
        self.offset = None
        self.precursor_mz = []
        self.precursor_charge = []
        self.scan_id = []
        self.mz_arrays = []
        self.intensity_arrays = []
        self.annotations = None

    @abstractmethod
    def open(self) -> Iterable:
        """Open the file as an iterable."""

    @abstractmethod
    def parse_spectrum(self, spectrum: dict) -> MassSpectrum | None:
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in a given format.

        Returns
        -------
        MassSpectrum or None
            The parsed mass spectrum or None if it is skipped.
        """

    def read(self) -> BaseParser:
        """Read the ms data file.

        Returns
        -------
        Self
        """
        n_skipped = 0
        with self.open() as spectra:
            for spectrum in tqdm(spectra, desc=str(self.path), unit="spectra"):
                try:
                    spectrum = self.parse_spectrum(spectrum)
                    if spectrum is None:
                        continue

                    if self.preprocessing_fn is not None:
                        for processor in self.preprocessing_fn:
                            spectrum = processor(spectrum)

                    self.mz_arrays.append(spectrum.mz)
                    self.intensity_arrays.append(spectrum.intensity)
                    self.precursor_mz.append(spectrum.precursor_mz)
                    self.precursor_charge.append(spectrum.precursor_charge)
                    self.scan_id.append(_parse_scan_id(spectrum.scan_id))
                    if self.annotations is not None:
                        self.annotations.append(spectrum.label)
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
    def n_spectra(self) -> int:
        """The number of spectra."""
        return self.offset.shape[0]

    @property
    def n_peaks(self) -> int:
        """The number of peaks in the file."""
        return self.mz_arrays.shape[0]


class MzmlParser(BaseParser):
    """Parse mass spectra from an mzML file.

    Parameters
    ----------
    ms_data_file : PathLike
        The mzML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    """

    def open(self) -> Iterable[dict]:
        """Open the mzML file for reading."""
        return MzML(str(self.path))

    def parse_spectrum(self, spectrum: dict) -> MassSpectrum | None:
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in mzML format.

        Returns
        -------
        MassSpectrum or None
            The parsed mass spectrum or None if not at the correct MS level.
        """
        if spectrum["ms level"] != self.ms_level:
            return None

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
            return MassSpectrum(
                filename=str(self.path),
                scan_id=spectrum["id"],
                mz=spectrum["m/z array"],
                intensity=spectrum["intensity array"],
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
            )

        raise ValueError("Invalid precursor charge")


class MzxmlParser(BaseParser):
    """Parse mass spectra from an mzXML file.

    Parameters
    ----------
    ms_data_file : PathLike
        The mzXML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    """

    def open(self) -> Iterable[dict]:
        """Open the mzXML file for reading."""
        return MzXML(str(self.path))

    def parse_spectrum(self, spectrum: dict) -> MassSpectrum | None:
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in mzXML format.

        Returns
        -------
        MassSpectrum
            The parsed mass spectrum.
        """
        if spectrum["msLevel"] != self.ms_level:
            return None

        if self.ms_level > 1:
            precursor = spectrum["precursorMz"][0]
            precursor_mz = float(precursor["precursorMz"])
            precursor_charge = int(precursor.get("precursorCharge", 0))
        else:
            precursor_mz, precursor_charge = None, 0

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            return MassSpectrum(
                filename=str(self.path),
                scan_id=spectrum["id"],
                mz=spectrum["m/z array"],
                intensity=spectrum["intensity array"],
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
            )

        raise ValueError("Invalid precursor charge")


class MgfParser(BaseParser):
    """Parse mass spectra from an MGF file.

    Parameters
    ----------
    ms_data_file : PathLike
        The MGF file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    annotations : bool
        Include peptide annotations.
    """

    def __init__(
        self,
        ms_data_file: PathLike,
        ms_level: int = 2,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        annotations: bool = False,
    ) -> None:
        """Initialize the MgfParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            id_type="index",
        )
        if annotations:
            self.annotations = []

        self._counter = -1

    def open(self) -> Iterable[dict]:
        """Open the MGF file for reading."""
        return MGF(str(self.path))

    def parse_spectrum(self, spectrum: dict) -> MassSpectrum:
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
            label = spectrum["params"].get("seq")
        else:
            label = None

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            return MassSpectrum(
                filename=str(self.path),
                scan_id=self._counter,
                mz=spectrum["m/z array"],
                intensity=spectrum["intensity array"],
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                label=label,
            )

        raise ValueError("Invalid precursor charge")


def _parse_scan_id(scan_str: str | int) -> int:
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
