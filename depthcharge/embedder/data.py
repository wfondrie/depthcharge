"""Our pytorch Dataset classes"""
import logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from pyteomics.mzml import MzML

from . import similarity
from .. import utils

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class SpectrumDataset(torch.utils.data.IterableDataset):
    """Parse and retrieve collections of mass spectra.

    Parameters
    ----------
    mzml_files : str, Path object, or List of either.
        One or more mzML files to parse. Alternatively, these can be the
        cached *.npy files.
    n_peaks : int, optional
        Keep only the top-n most instense peaks in any spectrum. :code:`None`
        retains all of the peaks.
    n_pairs : int, optional
        The number of randomly paired spectra to use for evaluation.
    frac_identical : float, optional
        The fraction of pairs that should consist of the same spectrum during
        training.
    rng : int or RandomState, optional
        The numpy random number generator.
    ms_level : int, optional
        The MSn level to analyze.
    tol : float, optional
        The tolerance with which to calculate the generalized scalar product.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    cache_dir : str or Path, optional
        The directory in which to save the cached spectra.
    file_root : str or list of str
        The prefix to add to the cached files.
    """

    def __init__(
        self,
        mzml_files,
        n_peaks=200,
        n_pairs=10000,
        frac_identical=0.0,
        rng=42,
        ms_level=2,
        tol=0.01,
        min_mz=140,
        cache_dir=None,
        file_root=None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = int(n_peaks)
        self.n_pairs = int(n_pairs)
        self.tol = float(tol)
        self.min_mz = float(min_mz)
        self.frac_identical = float(frac_identical)
        self.cache_dir = cache_dir
        self.rng = rng
        self._ms_level = int(ms_level)

        # Parse the spectra
        self._file_map, self._spectra = _parse_spectra(
            mzml_files=mzml_files,
            ms_level=self.ms_level,
            npy_dir=self.cache_dir,
            prefix=file_root,
        )

        # Setup for evaluation:
        self._eval = False
        self._eval_pairs = self.rng.integers(
            self.n_spectra, size=(self.n_pairs, 2)
        )

    def __len__(self):
        """The number of spectra."""
        if self._eval:
            return len(self._eval_pairs)

        raise TypeError("'len()' is not defined in training mode.")

    def __iter__(self):
        """Generate random pairs."""
        if self._eval:
            for idx in range(len(self._eval_pairs)):
                yield self[idx]

        else:
            while True:
                yield self[0]

    def __getitem__(self, idx):
        """Return a single spectrum-pair"""
        if self._eval:
            x_idx, y_idx = self._eval_pairs[idx, :]
        else:
            x_idx, y_idx = self.rng.integers(self.n_spectra, size=2)
            if self.rng.binomial(1, p=self.frac_identical):
                y_idx = x_idx

        x_spec = self._get_spectrum(x_idx)
        y_spec = self._get_spectrum(y_idx)
        gsp = similarity.gsp(x_spec, y_spec, self.tol)
        return torch.tensor(x_spec), torch.tensor(y_spec), gsp

    def _get_spectrum(self, idx):
        """Retrieve a spectrum"""
        sfile, sloc = self._spectra[idx]
        indices, specs = self._file_map[sfile]
        spec = specs[indices[sloc] : indices[sloc + 1], :]
        return self._process_peaks(spec)

    def _process_peaks(self, spec):
        """Choose the top n peaks and normalize the spectrum intensities."""
        mz_array, int_array = spec.T
        if self.min_mz is not None:
            keep = mz_array >= self.min_mz
            mz_array = mz_array[keep]
            int_array = int_array[keep]

        if len(int_array) > self.n_peaks:
            top_p = np.argpartition(int_array, -self.n_peaks)[-self.n_peaks :]
            top_p = np.sort(top_p)
            mz_array = mz_array[top_p]
            int_array = int_array[top_p]

        int_array = np.sqrt(int_array)
        int_array = int_array / np.linalg.norm(int_array)
        return np.array([mz_array, int_array]).T

    def train(self):
        """Set the dataset to training mode"""
        self._eval = False

    def eval(self):
        """Set the datset to validation mode"""
        self._eval = True

    @property
    def n_spectra(self):
        """The total number of spectra."""
        return len(self._spectra)

    @property
    def files(self):
        """The cached spectrum files."""
        return list(self._file_map.keys())

    @property
    def ms_level(self):
        """The MSn level to analyze."""
        return self._ms_level

    @property
    def rng(self):
        """The numpy random number generator"""
        return self._rng

    @rng.setter
    def rng(self, rng):
        """Set the random number generators"""
        self._rng = np.random.default_rng(rng)

    @property
    def cache_dir(self):
        """The directory where the parsed mass spectra are cached"""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, path):
        """Set the cache directory"""
        if path is None:
            self._cache_dir = Path.cwd()
        else:
            self._cache_dir = Path(path)

        if not self._cache_dir.exists():
            raise FileNotFoundError(
                f"The requested cache directory (self._cache_dir) does not "
                "exist."
            )


# Utility functions -----------------------------------------------------------
def _parse_spectra(mzml_files, ms_level, npy_dir, prefix):
    """Read the MS2 spectra from a series of mzML files."""
    mzml_files = utils.listify(mzml_files)
    prefix = utils.listify(prefix)

    if len(prefix) == 1:
        prefix = prefix * len(mzml_files)

    # Read spectra from mzML files:
    file_map = {}
    all_spectra = []
    for idx, (mzml_file, pre) in enumerate(zip(tqdm(mzml_files), prefix)):
        if not str(mzml_file).endswith(".npy"):
            npy_file, idx_file = _parse_mzml(
                mzml_file=mzml_file,
                ms_level=ms_level,
                npy_dir=npy_dir,
                prefix=pre,
            )
        else:
            npy_file, idx_file = _parse_npy(mzml_file)

        file_map[idx] = (np.load(idx_file), np.load(npy_file, mmap_mode="r"))
        all_spectra += [[idx, s] for s in range(len(file_map[idx][0]) - 1)]

    return file_map, np.array(all_spectra)


def _parse_npy(npy_file):
    """Verify that a npy file can be matched to it's index"""
    exts = Path(npy_file).suffixes
    if exts[-2:] == [".index", ".npy"]:
        idx_file = Path(npy_file)
        npy_file = Path(str(idx_file).replace(".index.npy", ".npy"))
    else:
        npy_file = Path(npy_file)
        idx_file = Path(str(npy_file).replace(".npy", ".index.npy"))

    if not npy_file.exists():
        FileNotFoundError(
            f"The cached spectrum file could not be found: {npy_file}"
        )

    if not idx_file.exists():
        FileNotFoundError(
            "The index for the cached spectrum file could not be found: "
            f"{idx_file}"
        )

    return npy_file, idx_file


def _parse_mzml(mzml_file, ms_level, npy_dir, prefix):
    """If we need to actually parse the mzml file"""
    root = Path(mzml_file).stem
    if prefix is not None:
        root = str(prefix) + root

    npy_file = Path(npy_dir, f"{root}.ms{ms_level}.npy")
    idx_file = Path(npy_dir, f"{root}.ms{ms_level}.index.npy")

    # Parse from the mzML file only if they don't already exist:
    if not (npy_file.exists() and idx_file.exists()):
        with MzML(str(mzml_file)) as mzml_ref:
            spectra = [
                _parse_spectrum(s)
                for s in tqdm(mzml_ref)
                if s["ms level"] == ms_level
            ]

            _save_spectra(spectra, npy_file, idx_file)
            del spectra

    return npy_file, idx_file


def _parse_spectrum(spectrum, ms_level=2):
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
        return None

    mz = spectrum["m/z array"].astype(np.float32)
    intensity = spectrum["intensity array"].astype(np.float32)
    intensity = intensity / np.linalg.norm(intensity)
    return np.array([mz, intensity]).T


def _save_spectra(spectra, spec_file, index_file):
    """Save spectra"""
    indices = np.array([0] + [s.shape[0] for s in spectra]).cumsum()
    spec_array = np.vstack(spectra)
    np.save(index_file, indices)
    np.save(spec_file, spec_array)
