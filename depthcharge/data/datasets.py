"""A PyTorch Dataset class for annotated spectra."""
import logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from pyteomics.mzml import MzML
from pyteomics.mgf import MGF

from . import parse
from .. import utils
from .. import similarity

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class SpectrumDataset(torch.utils.data.IterableDataset):
    """Parse and retrieve collections of mass spectra

    Parameters
    ----------
    ms_data_files : str, Path object, or List of either
        One or more mzML or MGF files containing mass spectra. Alternatively
        these can be the cached *.npy files.
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. :code:`None`
        retains all of the peaks.
    ms_level : int, optional,
        The MSn level to analyze.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    rng : int or RandomState, optional.
        The numpy random state. :code:`None` leaves mass spectra in the order
        they were parsed.
    cache_dir : str or Path, optional
        The directory in which to save the cached spectra.
    file_root : str or list of str
        The prefix to add to the cached files.
    """

    def __init__(
        self,
        ms_data_files,
        n_peaks=200,
        ms_level=2,
        min_mz=140,
        rng=None,
        cache_dir=None,
        file_root=None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = int(n_peaks)
        self.min_mz = float(min_mz)
        self.rng = rng
        self.cache_dir = cache_dir
        self._ms_level = int(ms_level)

        try:
            self._require_prec
        except AttributeError:
            self._require_prec = True

        try:
            self._peptides
        except AttributeError:
            self._peptides = False

        self._file_map, self._spectra = parse.spectra(
            ms_data_files=ms_data_files,
            ms_level=self.ms_level,
            npy_dir=self.cache_dir,
            prefix=file_root,
            require_prec=self._require_prec,
            peptides=self._peptides,
        )

        self._order = np.arange(len(self))
        if self.rng is not None:
            self.rng.shuffle(self._order)

    def __len__(self):
        """The number of spectra"""
        return self.n_spectra

    def __iter__(self):
        """Return spectra"""
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        """Return a single mass spectrum"""
        spec_idx = self._order[idx]
        spec, prec, pep = self._get_spectrum(spec_idx)
        return torch.tensor(spec), prec, pep

    def _get_spectrum(self, idx):
        """Retrieve a spectrum"""
        sfile, sloc = self._spectra[idx]
        indices, specs, precursors, peptides = self._file_map[sfile]
        spec = specs[indices[sloc] : indices[sloc + 1], :]
        if precursors is not None:
            precursor = tuple(precursors[sloc, :])
        else:
            precursor = (None, None)

        if peptides is not None:
            peptide = peptides[sloc]
        else:
            peptide = None

        return self._process_peaks(spec), precursor, peptide

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
        if rng is not None:
            self._rng = np.random.default_rng(rng)
        else:
            self._rng = None

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


class PairedSpectrumDataset(SpectrumDataset):
    """Parse and retrieve collections of mass spectra.

    Parameters
    ----------
    ms_data_files : str, Path object, or List of either.
        One or more mzML or MGF files to parse. Alternatively, these can be the
        cached *.npy files.
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. :code:`None`
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
        ms_data_files,
        n_peaks=200,
        n_pairs=10000,
        p_identical=0.0,
        ms_level=2,
        min_mz=140,
        tol=0.01,
        rng=42,
        cache_dir=None,
        file_root=None,
    ):
        """Initialize a PairedSpectrumDataset"""
        self._require_prec = False
        super().__init__(
            ms_data_files=ms_data_files,
            n_peaks=n_peaks,
            ms_level=ms_level,
            min_mz=min_mz,
            rng=rng,
            cache_dir=cache_dir,
            file_root=file_root,
        )
        self.n_pairs = int(n_pairs)
        self.tol = float(tol)
        self.p_identical = float(p_identical)

        # Setup for evaluation
        self._eval = False
        self._eval_pairs = self.rng.integers(
            self.n_spectra, size=(self.n_pairs, 2)
        )

    def __len__(self):
        """The number of pairs"""
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
            if self.rng.binomial(1, p=self.p_identical):
                y_idx = x_idx

        x_spec, _, _ = self._get_spectrum(x_idx)
        y_spec, _, _ = self._get_spectrum(y_idx)
        sim = similarity.gsp(x_spec, y_spec, self.tol)
        return torch.tensor(x_spec), torch.tensor(y_spec), sim

    @property
    def rng(self):
        """The random number generator"""
        return self._rng

    @rng.setter
    def rng(self, rng):
        """Set the random number generators"""
        self._rng = np.random.default_rng(rng)


class AnnotatedSpectrumDataset(SpectrumDataset):
    """Parse and retrieve collections of mass spectra

    Parameters
    ----------
    mgf_files : str, Path object, or List of either
        One or more MGF files containing mass spectra. Alternatively
        these can be the cached *.npy files.
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. :code:`None`
        retains all of the peaks.
    ms_level : int, optional,
        The MSn level to analyze.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    rng : int or RandomState, optional.
        The numpy random state. :code:`None` leaves mass spectra in the order
        they were parsed.
    cache_dir : str or Path, optional
        The directory in which to save the cached spectra.
    file_root : str or list of str
        The prefix to add to the cached files.
    """

    def __init__(
        self,
        mgf_files,
        n_peaks=200,
        ms_level=2,
        min_mz=140,
        rng=None,
        cache_dir=None,
        file_root=None,
    ):
        """Initialize a SpectrumDataset"""
        self._peptides = True
        super().__init__(
            ms_data_files=mgf_files,
            n_peaks=n_peaks,
            ms_level=ms_level,
            min_mz=min_mz,
            rng=rng,
            cache_dir=cache_dir,
            file_root=file_root,
        )
