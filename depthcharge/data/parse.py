"""Utility functions for parsing mass spectra."""
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm
from pyteomics.mzml import MzML
from pyteomics.mgf import MGF

from .. import utils

LOGGER = logging.getLogger(__name__)


def spectra(
    ms_data_files, ms_level, npy_dir, prefix, require_prec=True, peptides=False
):
    """Read the MS2 spectra from a series of mzML files."""
    ms_data_files = utils.listify(ms_data_files)
    prefix = utils.listify(prefix)

    if len(prefix) == 1:
        prefix = prefix * len(ms_data_files)

    # Read spectra from mzML files:
    file_map = {}
    all_spectra = []
    for idx, (ms_file, pre) in enumerate(zip(tqdm(ms_data_files), prefix)):
        if not str(ms_file).lower().endswith(".npy"):
            npy_file, idx_file, prec_file, pep_file = _parse_spec_data(
                ms_file=ms_file,
                ms_level=ms_level,
                npy_dir=npy_dir,
                prefix=pre,
                require_prec=require_prec,
                peptides=peptides,
            )
        else:
            npy_file, idx_file, prec_file, pep_file = _parse_npy(
                ms_file, require_prec=require_prec, peptides=peptides
            )

        data = [
            np.load(idx_file),
            np.load(npy_file, mmap_mode="r"),
            None,
            None,
        ]

        if prec_file is not None:
            data[2] = np.load(prec_file)

        if pep_file is not None:
            data[3] = np.load(pep_file)

        file_map[idx] = tuple(data)
        all_spectra += [[idx, s] for s in range(len(file_map[idx][0]) - 1)]

    return file_map, np.array(all_spectra)


def _parse_npy(npy_file, require_prec=True, peptides=False):
    """Verify that a npy file can be matched to it's index"""
    exts = Path(npy_file).suffixes
    if exts[-2:] == [".index", ".npy"]:
        idx_file = Path(npy_file)
        npy_file = Path(str(idx_file).replace(".index.npy", ".npy"))
    else:
        npy_file = Path(npy_file)
        idx_file = Path(str(npy_file).replace(".npy", ".index.npy"))

    if not npy_file.exists():
        raise FileNotFoundError(
            f"The cached spectrum file could not be found: {npy_file}"
        )

    if not idx_file.exists():
        raise FileNotFoundError(
            "The index for the cached spectrum file could not be found: "
            f"{idx_file}"
        )

    prec_file = Path(str(npy_file).replace(".npy", ".precursors.npy"))
    if not prec_file.exists():
        if require_prec:
            raise FileNotFoundError(
                "The precursor masses for the cached spectrum file could not "
                f"be found: {prec_file}"
            )

        prec_file = None

    pep_file = Path(str(npy_file).replace(".npy", ".peptides.npy"))
    if not pep_file.exists():
        if peptides:
            raise FileNotFoundError(
                "The peptides for the cached spectrum file could not be found:"
                f"{pep_file}"
            )

        pep_file = None

    return npy_file, idx_file, prec_file, pep_file


def _parse_spec_data(
    ms_file, ms_level, npy_dir, prefix, require_prec=True, peptides=False
):
    """If we need to actually parse the mzml file"""
    pep_file = None
    prec_file = None
    root = Path(ms_file).stem
    if prefix is not None:
        root = str(prefix) + root

    npy_file = Path(npy_dir, f"{root}.ms{ms_level}.npy")
    idx_file = Path(npy_dir, f"{root}.ms{ms_level}.index.npy")
    all_files = [npy_file, idx_file]
    if ms_level != 1 and require_prec:
        prec_file = Path(npy_dir, f"{root}.ms{ms_level}.precursors.npy")
        all_files.append(prec_file)

    if peptides:
        pep_file = Path(npy_dir, f"{root}.ms{ms_level}.peptides.npy")
        all_files.append(pep_file)

    # Parse from the mzML file only if they don't already exist:
    if not all(f.exists() for f in all_files):
        if str(ms_file).lower().endswith(".mzml") and not peptides:
            opener = MzML
            parse_fun = _parse_mzml_spectrum
        elif str(ms_file).lower().endswith(".mgf"):
            opener = MGF
            parse_fun = _parse_mgf_spectrum

        with opener(str(ms_file)) as ms_ref:
            spec = [parse_fun(s, ms_level, peptides) for s in tqdm(ms_ref)]
            _save_spectra(spec, npy_file, idx_file, prec_file, pep_file)
            del spec

    return npy_file, idx_file, prec_file, pep_file


def _parse_mzml_spectrum(spectrum, ms_level=2, *args, **kwargs):
    """Read an MS2 spectrum from an MzML file

    Parameters
    ----------
    spectrum_data : Dict
        The dictionary from pyteomics.
    n_peaks : int, optional
        The maximum number of peaks to keep (in order of intensity).

    Returns
    -------
    torch.Tensor or None
    """
    if spectrum["ms level"] != ms_level:
        return None, None

    if ms_level > 1:
        data = spectrum["precursorList"]["precursor"][0]["selectedIonList"][
            "selectedIon"
        ][0]
        precursor = np.array([data["selected ion m/z"], data["charge state"]])
        precursor = precursor.astype(np.float32).T
    else:
        precursor = None

    mz = spectrum["m/z array"].astype(np.float32)
    intensity = spectrum["intensity array"].astype(np.float32)
    return np.array([mz, intensity]).T, precursor, None


def _parse_mgf_spectrum(spectrum, ms_level=2, peptides=False, *args, **kwargs):
    """Read an MS2 spectrum from an MGF file

    Parameters
    ----------
    spectrum_data : Dict
        The dictionary from pyteomics.
    n_peaks : int, optional
        The maximum number of peaks to keep (in order of intensity).

    Returns
    -------
    torch.Tensor or None
    """
    pep = None
    if peptides:
        pep = spectrum["params"]["seq"]

    prec_mz = spectrum["params"]["pepmass"][0]
    prec_charge = spectrum["params"]["charge"][0]
    mz = spectrum["m/z array"].astype(np.float32)
    intensity = spectrum["intensity array"].astype(np.float32)
    return np.array([mz, intensity]).T, np.array([prec_mz, prec_charge]).T, pep


def _save_spectra(spec, spec_file, index_file, prec_file, pep_file):
    """Save spectra"""
    spec = [s for s in spec if s[0] is not None]
    spec, prec, pep = list(zip(*spec))
    indices = np.array([0] + [s.shape[0] for s in spec]).cumsum()
    np.save(index_file, indices)
    np.save(spec_file, np.vstack(spec))
    if all([p is not None for p in prec]):
        np.save(prec_file, np.vstack(prec))

    if pep_file is not None:
        np.save(pep_file, np.array(pep))
