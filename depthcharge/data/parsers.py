"""Mass spectrometry data parsers."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path
from typing import Any

import pyarrow as pa
from pyteomics.mgf import MGF
from pyteomics.mzml import MzML
from pyteomics.mzxml import MzXML
from tqdm.auto import tqdm

from .. import utils
from ..primitives import MassSpectrum
from . import preprocessing

LOGGER = logging.getLogger(__name__)


class BaseParser(ABC):
    """A base parser class to inherit from.

    Parameters
    ----------
    peak_file : PathLike
        The peak file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    custom_fields : dict of str to list of str, optional
        Additional field to extract during peak file parsing. The key must
        be the resulting column name and value must be an interable of
        containing the necessary keys to retreive the value from the
        spectrum from the corresponding Pyteomics parser.
    id_type : str, optional
        The Hupo-PSI prefix for the spectrum identifier.
    """

    def __init__(
        self,
        peak_file: PathLike,
        ms_level: int | Iterable[int] | None = 2,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        custom_fields: dict[str, str | Iterable[str]] | None = None,
        id_type: str = "scan",
    ) -> None:
        """Initialize the BaseParser."""
        self.peak_file = Path(peak_file)
        self.ms_level = (
            ms_level if ms_level is None else set(utils.listify(ms_level))
        )

        if preprocessing_fn is None:
            self.preprocessing_fn = [
                preprocessing.set_mz_range(min_mz=140),
                preprocessing.filter_intensity(max_num_peaks=200),
                preprocessing.scale_intensity(scaling="root"),
                preprocessing.scale_to_unit_norm,
            ]
        else:
            self.preprocessing_fn = utils.listify(preprocessing_fn)

        self.valid_charge = None if valid_charge is None else set(valid_charge)
        self.custom_fields = custom_fields
        self.id_type = id_type

        # Check format:
        self.sniff()

    @abstractmethod
    def sniff(self) -> None:
        """Quickly test a file for the correct type.

        Raises
        ------
        IOError
            Raised if the file is not the expected format.
        """

    @abstractmethod
    def open(self) -> Iterable[dict]:
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

    def parse_custom_fields(self, spectrum: dict) -> dict[str, Any]:
        """Parse user-provided fields.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in a given format.

        Returns
        -------
        dict
            The parsed value of each, whatever it may be.
        """
        out = {}
        if self.custom_fields is None:
            return out

        for field, accessors in self.custom_fields.items():
            val = spectrum
            for accessor in accessors:
                try:
                    val = val[accessor]
                except (KeyError, IndexError) as exc:
                    raise KeyError(
                        f"'{accessor}' not found in {accessors} for custom field"
                        f" '{field}'. Current state = {val}"
                    ) from exc

            out[field] = val

        return out

    def iter_batches(self, batch_size: int | None) -> pa.RecordBatch:
        """Iterate over batches of mass spectra in the Arrow format.

        Parameters
        ----------
        batch_size : int or None
            The number of spectra in a batch. ``None`` loads all of
            the spectra in a single batch.

        Yields
        ------
        RecordBatch
            A batch of spectra and their metadata.
        """
        batch_size = float("inf") if batch_size is None else batch_size
        metadata = {
            "peak_file": self.peak_file.name,
            "id_type": self.id_type,
        }

        pbar_args = {
            "desc": self.peak_file.name,
            "unit": " spectra",
        }

        n_skipped = 0
        with self.open() as spectra:
            batch = []
            for spectrum in tqdm(spectra, **pbar_args):
                try:
                    parsed = self.parse_spectrum(spectrum)
                    if parsed is None:
                        continue

                    if self.preprocessing_fn is not None:
                        for processor in self.preprocessing_fn:
                            parsed = processor(parsed)

                    entry = {
                        "peak_file": self.peak_file.name,
                        "scan_id": _parse_scan_id(parsed.scan_id),
                        "ms_level": parsed.ms_level,
                        "precursor_mz": parsed.precursor_mz,
                        "precursor_charge": parsed.precursor_charge,
                        "mz_array": parsed.mz,
                        "intensity_array": parsed.intensity,
                    }

                except (IndexError, KeyError, ValueError):
                    n_skipped += 1
                    continue

                # Parse custom fields:
                entry.update(self.parse_custom_fields(spectrum))
                batch.append(entry)

                if len(batch) >= batch_size:
                    yield pa.RecordBatch.from_pylist(batch, metadata=metadata)
                    batch = []

            # Get the remainder:
            if batch:
                yield pa.RecordBatch.from_pylist(batch, metadata=metadata)

        if n_skipped:
            LOGGER.warning(
                "Skipped %d spectra with invalid information", n_skipped
            )


class MzmlParser(BaseParser):
    """Parse mass spectra from an mzML file.

    Parameters
    ----------
    peak_file : PathLike
        The mzML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    custom_fields : dict of str to list of str, optional
        Additional field to extract during peak file parsing. The key must
        be the resulting column name and value must be an interable of
        containing the necessary keys to retreive the value from the
        spectrum from the corresponding Pyteomics parser.
    """

    def sniff(self) -> None:
        """Quickly test a file for the correct type.

        Raises
        ------
        IOError
            Raised if the file is not the expected format.
        """
        with self.peak_file.open() as mzdat:
            next(mzdat)
            if "http://psi.hupo.org/ms/mzml" not in next(mzdat):
                raise OSError("Not an mzML file.")

    def open(self) -> Iterable[dict]:
        """Open the mzML file for reading."""
        return MzML(str(self.peak_file))

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
        ms_level = spectrum["ms level"]
        if self.ms_level is not None and ms_level not in self.ms_level:
            return None

        if ms_level > 1:
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
                filename=str(self.peak_file),
                scan_id=spectrum["id"],
                mz=spectrum["m/z array"],
                intensity=spectrum["intensity array"],
                ms_level=ms_level,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
            )

        raise ValueError("Invalid precursor charge.")


class MzxmlParser(BaseParser):
    """Parse mass spectra from an mzXML file.

    Parameters
    ----------
    peak_file : PathLike
        The mzXML file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    custom_fields : dict of str to list of str, optional
        Additional field to extract during peak file parsing. The key must
        be the resulting column name and value must be an interable of
        containing the necessary keys to retreive the value from the
        spectrum from the corresponding Pyteomics parser.
    """

    def sniff(self) -> None:
        """Quickly test a file for the correct type.

        Raises
        ------
        IOError
            Raised if the file is not the expected format.
        """
        scent = "http://sashimi.sourceforge.net/schema_revision/mzXML"
        with self.peak_file.open() as mzdat:
            next(mzdat)
            if scent not in next(mzdat):
                raise OSError("Not an mzXML file.")

    def open(self) -> Iterable[dict]:
        """Open the mzXML file for reading."""
        return MzXML(str(self.peak_file))

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
        ms_level = spectrum["msLevel"]
        if self.ms_level is not None and ms_level not in self.ms_level:
            return None

        if ms_level > 1:
            precursor = spectrum["precursorMz"][0]
            precursor_mz = float(precursor["precursorMz"])
            precursor_charge = int(precursor.get("precursorCharge", 0))
        else:
            precursor_mz, precursor_charge = None, 0

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            return MassSpectrum(
                filename=str(self.peak_file),
                scan_id=spectrum["id"],
                mz=spectrum["m/z array"],
                intensity=spectrum["intensity array"],
                ms_level=ms_level,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
            )

        raise ValueError("Invalid precursor charge")


class MgfParser(BaseParser):
    """Parse mass spectra from an MGF file.

    Parameters
    ----------
    peak_file : PathLike
        The MGF file to parse.
    ms_level : int
        The MS level of the spectra to parse.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    custom_fields : dict of str to list of str, optional
        Additional field to extract during peak file parsing. The key must
        be the resulting column name and value must be an interable of
        containing the necessary keys to retreive the value from the
        spectrum from the corresponding Pyteomics parser.
    """

    def __init__(
        self,
        peak_file: PathLike,
        ms_level: int = 2,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        custom_fields: dict[str, Iterable[str]] | None = None,
    ) -> None:
        """Initialize the MgfParser."""
        super().__init__(
            peak_file,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            custom_fields=custom_fields,
            id_type="index",
        )
        self._counter = -1
        self._assumed_ms_level = sorted(self.ms_level)[0]

    def sniff(self) -> None:
        """Quickly test a file for the correct type.

        Raises
        ------
        IOError
            Raised if the file is not the expected format.
        """
        with self.peak_file.open() as mzdat:
            if not next(mzdat).startswith("BEGIN IONS"):
                raise OSError("Not an MGF file.")

    def open(self) -> Iterable[dict]:
        """Open the MGF file for reading."""
        return MGF(str(self.peak_file))

    def parse_spectrum(self, spectrum: dict) -> MassSpectrum:
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in MGF format.
        """
        self._counter += 1
        if self.ms_level is not None and 1 not in self.ms_level:
            precursor_mz = float(spectrum["params"]["pepmass"][0])
            precursor_charge = int(spectrum["params"].get("charge", [0])[0])
        else:
            precursor_mz, precursor_charge = None, 0

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            return MassSpectrum(
                filename=str(self.peak_file),
                scan_id=self._counter,
                mz=spectrum["m/z array"],
                intensity=spectrum["intensity array"],
                ms_level=self._assumed_ms_level,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
            )

        raise ValueError("Invalid precursor charge.")


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


class ParserFactory:
    """Figure out what parser to use."""

    parsers = [
        MzmlParser,
        MzxmlParser,
        MgfParser,
    ]

    @classmethod
    def get_parser(cls, peak_file: PathLike, **kwargs: dict) -> BaseParser:
        """Get the correct parser for a peak file.

        Parameters
        ----------
        peak_file: PathLike
            The peak file to parse.
        kwargs : dict
            Keyword arguments to pass to the parser.
        """
        for parser in cls.parsers:
            try:
                return parser(peak_file, **kwargs)
            except OSError:
                pass

        raise OSError("Unknown file format.")
