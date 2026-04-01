"""Mass spectrometry data parsers."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from os import PathLike, fspath
from typing import Any
import numpy as np 
import pandas as pd 

import pyarrow as pa
import timsrust_pyo3
from cloudpathlib import AnyPath
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
    progress : bool, optional
        Enable or disable the progress bar.
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
        progress: bool = True,
        id_type: str = "scan",
    ) -> None:
        """Initialize the BaseParser."""
        self.peak_file = AnyPath(peak_file)
        self.progress = progress
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

        # Used during parsing:
        self._batch = None

        # Define the schema
        self.schema = pa.schema(
            [
                pa.field("peak_file", pa.string()),
                pa.field("scan_id", pa.string()),
                pa.field("ms_level", pa.uint8()),
                pa.field("precursor_mz", pa.float64()),
                pa.field("precursor_charge", pa.int16()),
                pa.field("mz_array", pa.list_(pa.float64())),
                pa.field("intensity_array", pa.list_(pa.float64())),
            ]
        )

        if self.custom_fields is not None:
            self.custom_fields = utils.listify(self.custom_fields)
            for field in self.custom_fields:
                self.schema = self.schema.append(
                    pa.field(field.name, field.dtype)
                )

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

        for field in self.custom_fields:
            out[field.name] = field.accessor(spectrum)

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
        pbar_args = {
            "desc": self.peak_file.name,
            "unit": " spectra",
            "disable": not self.progress,
        }

        n_skipped = 0
        last_exc = None
        with self.open() as spectra:
            self._batch = None
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
                        "scan_id": str(parsed.scan_id),
                        "ms_level": parsed.ms_level,
                        "precursor_mz": parsed.precursor_mz,
                        "precursor_charge": parsed.precursor_charge,
                        "mz_array": parsed.mz,
                        "intensity_array": parsed.intensity,
                    }

                except (IndexError, KeyError, ValueError) as exc:
                    last_exc = exc
                    n_skipped += 1
                    continue

                # Parse custom fields:
                entry.update(self.parse_custom_fields(spectrum))
                self._update_batch(entry)

                # Update the batch:
                if len(self._batch["scan_id"]) == batch_size:
                    yield self._yield_batch()

            # Get the remainder:
            if self._batch is not None:
                yield self._yield_batch()

        if n_skipped:
            warnings.warn(
                f"Skipped {n_skipped} spectra with invalid information."
                f"Last error was: \n {str(last_exc)}"
            )

    def _update_batch(self, entry: dict) -> None:
        """Update the batch.

        Parameters
        ----------
        entry : dict
            The elemtn to add.

        """
        if self._batch is None:
            self._batch = {k: [v] for k, v in entry.items()}
        else:
            for key, val in entry.items():
                self._batch[key].append(val)

    def _yield_batch(self) -> pa.RecordBatch:
        """Yield the batch."""
        out = pa.RecordBatch.from_pydict(self._batch, schema=self.schema)
        self._batch = None
        return out

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
    progress : bool, optional
        Enable or disable the progress bar.

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
            precursor = spectrum["precursorList"]["precursor"]
            if len(precursor) > 1:
                LOGGER.warning(
                    "More than one precursor found for spectrum %s. "
                    "Only the first will be retained.",
                    spectrum["id"],
                )

            precursor_ion = precursor[0]["selectedIonList"]["selectedIon"]
            if len(precursor_ion) > 1:
                LOGGER.warning(
                    "More than one selected ions found for spectrum %s. "
                    "Only the first will be retained.",
                    spectrum["id"],
                )

            precursor_ion = precursor_ion[0]
            precursor_mz = float(precursor_ion["selected ion m/z"])
            charge = precursor_ion.get("charge state")
            if charge is not None:
                precursor_charge = int(charge)
            else:
                charge = precursor_ion.get("possible charge state")
                if charge is not None:
                    precursor_charge = int(charge)
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


class DiaParser(MzmlParser):
    def __init__(
        self,
        peak_file: PathLike,
        annotation_file: PathLike, 
        scan_width: int,
        ms_level: int = 2,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        custom_fields: dict[str, Iterable[str]] | None = None,
        progress: bool = True,
    ) -> None:
        """Initialize the DiaParser."""
        super().__init__(
            peak_file,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            custom_fields=custom_fields,
            progress=progress,
        )
        if annotation_file is None:
            raise ValueError("DiaParsers must have annotation files")
        
        self.anns = pd.read_csv(annotation_file, sep="\t")
        self.scan_width = scan_width

    def parse_spectrum(self, spectrum: dict) -> MassSpectrum: 
        scans = np.array(spectrum['scans'], dtype=object)
        rts = np.array(spectrum['rts'])
        ms1_scans = np.array(spectrum['ms1_scans'], dtype=object)
        ms1_rts = np.array(spectrum['ms1_rts'])

        abs_rts = [np.abs(x) for x in rts]
        sorted_rt_idxs = np.argsort(abs_rts)[:self.scan_width]
        rts = rts[sorted_rt_idxs]
        scans = scans[sorted_rt_idxs]

        abs_ms1_rts = [np.abs(x) for x in ms1_rts]
        sorted_ms1_rt_idxs = np.argsort(abs_ms1_rts)[:self.scan_width]
        ms1_rts = ms1_rts[sorted_ms1_rt_idxs]
        ms1_scans = ms1_scans[sorted_ms1_rt_idxs]

        annotations = {}
        annotations["window_width"] = spectrum['window_width']
        annotations["window_size"] = spectrum['window_size']
        annotations["rts"] = list(rts) + list(ms1_rts)
        for ann in spectrum["annotations"]: 
            annotations[ann] = spectrum["annotations"][ann]
        
        mzs = []
        intensities = []
        scan_numbers = []
        rt_values = []
        for scan, cur_rt in zip(scans,rts):
            for mz, intensity in scan: 
                mzs.append(mz)
                intensities.append(intensity) 
                scan_numbers.append(2)
                rt_values.append(cur_rt)
        
        for scan, cur_rt in zip(ms1_scans, ms1_rts):
            for mz1, intensity1 in scan: 
                mzs.append(mz1)
                intensities.append(intensity1)
                scan_numbers.append(1) 
                rt_values.append(cur_rt)

        annotations["scan_numbers"] = scan_numbers
        annotations["rts"] = rt_values

        filtered = self.anns.loc[self.anns['scan'] == spectrum["id"]]

        label = None
        if "sequence" in filtered.columns:
            label = filtered["sequence"].iloc[0]

        charge = spectrum.get("charge", None)
        if "charge" in filtered.columns:
            charge = filtered["charge"].iloc[0]
            
        rt = spectrum.get("retention_time", None)
        if "rt" in filtered.columns:
            rt = filtered["rt"].iloc[0]
            
        return MassSpectrum(
            filename=str(self.peak_file),
            scan_id=spectrum["id"],
            mz=mzs,
            intensity=intensities,
            retention_time=rt,
            precursor_mz=spectrum["precursor_m/z"],
            annotations=annotations,
            label=label,
            precursor_charge=charge,
        )

    @contextmanager
    def open(self) -> Iterable[dict]:
        f_to_mzrt_to_pep, max_mz, window_size, cycle_time = self.get_centers()
        precs_to_spec = []
        for part in f_to_mzrt_to_pep.keys():
            precs_to_spec.append(self.extract_spectra(f_to_mzrt_to_pep, part, (self.scan_width + 1) * cycle_time, max_mz, window_size))

        def _iter():
            for set in precs_to_spec: 
                for key, spec in set.items():
                    if 'ms1_scans' not in spec:
                        warnings.warn("Spectra does not have a MS1 scan. DiaParser requires MS1 scans."); continue
                    prec, rt, charge = key
                    spec["precursor_m/z"] = prec
                    spec["retention_time"] = rt
                    spec["charge"] =  charge 

                    yield spec 

        yield _iter()

    def get_centers(self):
        f_to_mzrt_to_pep = {}
        max_mz = 0
        num_spectra = 0
        part = 0
        last_rt = 0
        cycle_time = None
        with MzML(str(self.peak_file)) as reader:
            for spec in reader:
                if spec['ms level'] == 1:
                    cur_rt = 60 * spec['scanList']['scan'][0]['scan start time']
                    cycle_time = cur_rt - last_rt
                    last_rt = cur_rt
                if spec['ms level'] == 2:
                    window = spec['precursorList']['precursor'][0]['isolationWindow']
                    window_center = window['isolation window target m/z']
                    lower_offset = window['isolation window lower offset']
                    upper_offset = window['isolation window upper offset']
                    window_size = upper_offset + lower_offset
                    cur_rt = 60 * spec['scanList']['scan'][0]['scan start time']
                    if num_spectra % 50000 == 0:
                        part += 1
                        f_to_mzrt_to_pep[part] = {}
                    num_spectra += 1
                    key = (int(window_center/10), int(cur_rt/10))
                    max_mz = max(max_mz,int(window_center/10))
                    if key in f_to_mzrt_to_pep[part]:
                        f_to_mzrt_to_pep[part][key].append((window_center, cur_rt, 1))
                    else:
                        f_to_mzrt_to_pep[part][key] = [(window_center, cur_rt, 1)]
        return f_to_mzrt_to_pep, max_mz, window_size, cycle_time

    def extract_spectra(self, f_to_mzrt_to_pep, part, time_width, max_mz, window_size):
        prec_to_spec = {}
        n_skipped = 0
        last_exc = None
        with MzML(str(self.peak_file)) as reader:
            for spec in reader:
                cur_rt = 60 * spec['scanList']['scan'][0]['scan start time']
                if spec['ms level'] == 1:
                    for scan_rt in range(int(cur_rt/10) - 1, int(cur_rt/10) + 1):
                        for scan_window in range(max_mz+1):
                            if (scan_window, scan_rt) in f_to_mzrt_to_pep[part]:
                                for mz, rt, charge in f_to_mzrt_to_pep[part][(scan_window, scan_rt)]:
                                    if np.abs(rt - cur_rt) < time_width: 
                                        mzs = spec['m/z array']
                                        intensities = spec['intensity array']

                                        if (mz, rt, charge) not in prec_to_spec:
                                            prec_to_spec[(mz, rt, charge)] = {}
                                        if 'ms1_scans' not in prec_to_spec[(mz, rt, charge)]:
                                            prec_to_spec[(mz, rt, charge)]['ms1_scans'] = []
                                            prec_to_spec[(mz, rt, charge)]['ms1_rts'] = []
                                        prec_to_spec[(mz, rt, charge)]['ms1_scans'].append([x for x in zip(mzs, intensities)])
                                        prec_to_spec[(mz, rt, charge)]['ms1_rts'].append(cur_rt - rt)
                                        prec_to_spec[(mz, rt, charge)]['window_size'] = window_size
                elif spec['ms level'] == 2:
                    scan = spec["id"]
                    filtered = self.anns.loc[self.anns['scan'] == scan]
                    if len(filtered) > 1:
                        n_skipped += 1; last_exc = f"There are {len(filtered.columns)} spectra with {scan}. Scans must be unique"; continue 

                    window = spec['precursorList']['precursor'][0]['isolationWindow']
                    window_center = window['isolation window target m/z']
                    lower_offset = window['isolation window lower offset']
                    upper_offset = window['isolation window upper offset']
                
                    for scan_rt in range(int(cur_rt/10) - 1, int(cur_rt/10) + 1):
                        for scan_window in range(int((window_center - lower_offset)/10) - 1, int((window_center + upper_offset)/10) + 1):
                            if (scan_window, scan_rt) in f_to_mzrt_to_pep[part]:
                                for mz, rt, charge in f_to_mzrt_to_pep[part][(scan_window, scan_rt)]:
                                    in_mz = mz > window_center - lower_offset and mz < window_center + upper_offset
                                    rt_diff = np.abs(rt - cur_rt)
                                    if in_mz and rt_diff < time_width:
                                        mzs = spec['m/z array']
                                        intensities = spec['intensity array']

                                        if (mz, rt, charge) not in prec_to_spec:
                                            prec_to_spec[(mz, rt, charge)] = {}
                                        if 'annotations' not in prec_to_spec[(mz, rt, charge)]:
                                            prec_to_spec[(mz, rt, charge)]['annotations'] = {}
                                            prec_to_spec[(mz, rt, charge)]['id'] = scan
                                        if 'scans' not in prec_to_spec[(mz, rt, charge)]:
                                            prec_to_spec[(mz, rt, charge)]['scans'] = []
                                            prec_to_spec[(mz, rt, charge)]['rts'] = []
                                            prec_to_spec[(mz, rt, charge)]['window_width'] = max(lower_offset, upper_offset) 
                                        prec_to_spec[(mz, rt, charge)]['scans'].append([x for x in zip(mzs, intensities)])
                                        prec_to_spec[(mz, rt, charge)]['rts'].append(cur_rt - rt)
                                        row = filtered.iloc[0]
                                        for ann in filtered.columns:
                                            prec_to_spec[(mz, rt, charge)]['annotations'][ann] = row[ann]
        if n_skipped > 0: 
            warnings.warn(
                f"Skipped {n_skipped} spectra with invalid information."
                f"Last error was: \n {str(last_exc)}"
            )
        return prec_to_spec



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
    progress : bool, optional
        Enable or disable the progress bar.

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
    progress : bool, optional
        Enable or disable the progress bar.

    """

    def __init__(
        self,
        peak_file: PathLike,
        ms_level: int = 2,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        custom_fields: dict[str, Iterable[str]] | None = None,
        progress: bool = True,
    ) -> None:
        """Initialize the MgfParser."""
        super().__init__(
            peak_file,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            custom_fields=custom_fields,
            progress=progress,
            id_type="index",
        )
        self._counter = -1
        if ms_level is not None:
            self._assumed_ms_level = sorted(self.ms_level)[0]
        else:
            self._assumed_ms_level = None

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
                scan_id=f"index={self._counter}",
                mz=spectrum["m/z array"],
                intensity=spectrum["intensity array"],
                ms_level=self._assumed_ms_level,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
            )

        raise ValueError("Invalid precursor charge.")


class TdfParser(BaseParser):
    """Parse mass spectra from a TDF file.

    Parameters
    ----------
    peak_file : PathLike
        The TDF file to parse. Expects a *.d folder.
    ms_level : int
        The MS level of the spectra to parse. Currently supported: 2
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra.
    valid_charge : Iterable[int], optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    custom_fields : dict of str to list of str, optional
        Additional field to extract during peak file parsing. The key must
        be the resulting column name and value must be an interable of
        containing the necessary keys to retrieve the value from the
        spectrum from the corresponding Pyteomics parser.
    progress : bool, optional
        Enable or disable the progress bar.

    """

    def __init__(
        self,
        peak_file: PathLike,
        ms_level: int = 2,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        custom_fields: dict[str, Iterable[str]] | None = None,
        progress: bool = True,
    ) -> None:
        """Initialize the TdfParser."""
        warnings.warn(
            "Due to the current limitations of the timsrust library, "
            "m/z values are not temperature corrected for timsTOF files. "
            "This may lead to deviations in m/z values. In our experience "
            "these are typically mild, but use caution as your experience "
            "may vary depending on your instrument and settings."
        )

        if ms_level != 2 and ms_level is not None:
            raise ValueError(
                f"ms_level {ms_level} is currently not supported. "
                "Supported values are: 2."
            )
        if custom_fields is not None:
            warnings.warn(
                "custom_fields is currently not supported for "
                "TdfParser. Continuing with default value `None`.",
                UserWarning,
                stacklevel=2,
            )
            custom_fields = None
        super().__init__(
            peak_file,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            custom_fields=custom_fields,
            progress=progress,
            id_type="index",
        )
        self._counter = -1

    def sniff(self) -> None:
        """Quickly test a file for the correct type.

        Raises
        ------
        IOError
            Raised if the file is not the expected format.

        """
        if (
            not self.peak_file.exists()
            or self.peak_file.suffix.lower() != ".d"
        ):
            raise OSError("Not a TDF file.")
        try:
            timsrust_pyo3.SpectrumReader(fspath(self.peak_file))
        except OSError:
            raise OSError("Not a TDF file.")

    def _spectrum_to_dict(self, spectrum) -> dict:
        """Convert a Spectrum into a plain dict using the known schema."""
        p = spectrum.precursor
        """
        index: spectra are indexed by timsrust_pyo3 which aggregates multiple
        scans within a frame to single "spectra". NOT a Hupo PSI standard
        scan identifier. For more details, we refer to timsrust(_pyo3) code
        and documentation.
        """
        return {
            "index": spectrum.index,
            "mz_values": list(spectrum.mz_values),
            "intensities": list(spectrum.intensities),
            "precursor": {
                "mz": p.mz,
                "rt": p.rt,
                "im": p.im,
                "charge": p.charge,
                "intensity": p.intensity,
            },
            "collision_energy": spectrum.collision_energy,
            "isolation_mz": spectrum.isolation_mz,
            "isolation_width": spectrum.isolation_width,
        }

    @contextmanager
    def open(self):
        """Open the TDF file for reading."""
        reader = timsrust_pyo3.SpectrumReader(fspath(self.peak_file))
        broken_ids = []

        def _iter():
            n = len(reader)
            for i in range(n):
                try:
                    spec = reader.get(i)
                    yield self._spectrum_to_dict(spec)
                except OSError:
                    broken_ids.append(i)
                    continue  # skip broken spectra
            if len(broken_ids) > 0:
                warnings.warn(
                    "Skipped broken spectra at indices: "
                    + ", ".join(map(str, broken_ids))
                )

        yield _iter()

    def parse_spectrum(self, spectrum: dict) -> MassSpectrum:
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in TDF format.

        """
        self._counter += 1

        precursor_mz = float(spectrum["precursor"]["mz"])
        precursor_charge = float(spectrum["precursor"]["charge"])
        precursor_im = float(spectrum["precursor"]["im"])

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            return MassSpectrum(
                filename=str(self.peak_file),
                scan_id=spectrum[
                    "index"
                ],  # NOT a Hupo PSI standard scan identifier
                mz=spectrum["mz_values"],
                intensity=spectrum["intensities"],
                ms_level=2,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                ion_mobility=precursor_im,
            )

        raise ValueError("Invalid precursor charge.")


class ParserFactory:
    """Figure out what parser to use."""

    parsers = [MzmlParser, MzxmlParser, MgfParser, TdfParser]

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
        if "annotation_file" in kwargs:
            return DiaParser(peak_file, **kwargs)
        
        for parser in cls.parsers:
            try:
                return parser(peak_file, **kwargs)
            except OSError:
                pass

        raise OSError("Unknown file format.")
