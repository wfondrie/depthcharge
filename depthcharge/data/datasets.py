"""A PyTorch Dataset class for annotated spectra."""
import math

import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset

from .. import similarity


class SpectrumDataset(Dataset):
    """Parse and retrieve collections of mass spectra.

    Parameters
    ----------
    spectrum_index : depthcharge.data.SpectrumIndex
        The collection of spectra to use as a dataset.
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. ``None``
        retains all of the peaks.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    random_state : int or RandomState, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.

    Attributes
    ----------
    n_peaks : int
        The maximum number of mass speak to consider for each mass spectrum.
    min_mz : float
        The minimum m/z to consider for each mass spectrum.
    n_spectra : int
    index : depthcharge.data.SpectrumIndex
    rng : numpy.random.Generator
    """

    def __init__(
        self,
        spectrum_index,
        n_peaks=200,
        min_mz=140,
        random_state=None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self._rng = np.random.default_rng(random_state)
        self._index = spectrum_index
        self._order = np.arange(self.index.n_spectra)
        if self.rng is not None:
            self.rng.shuffle(self._order)

    def __len__(self):
        """The number of spectra."""
        return self.n_spectra

    def __getitem__(self, idx):
        """Return a single mass spectrum.

        Parameters
        ----------
        idx : int
            The index to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum[:, 0]`` are the m/z values and
            ``spectrum[:, 1]`` are their associated intensities.
        precursor_mz : float
            The m/z of the precursor.
        precursor_charge : int
            The charge of the precursor.
        """
        spec_idx = self._order[idx]
        mz_array, int_array, prec_mz, prec_charge = self.index[spec_idx]
        spec = self._process_peaks(mz_array, int_array)
        return spec, prec_mz, prec_charge

    def _process_peaks(self, mz_array, int_array):
        """Choose the top n peaks and normalize the spectrum intensities.

        Parameters
        ----------
        mz_array : numpy.ndarray of shape (n_peaks,)
            The m/z values of the peaks in the spectrum.
        int_array : numpy.ndarray of shape (n_peaks,)
            The intensity values of the peaks in the spectrum.

        Returns
        -------
        torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum[:, 0]`` are the m/z values and
            ``spectrum[:, 1]`` are their associated intensities.
        """
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
        return torch.tensor([mz_array, int_array]).T.float()

    @property
    def n_spectra(self):
        """The total number of spectra."""
        return self.index.n_spectra

    @property
    def index(self):
        """The underyling SpectrumIndex."""
        return self._index

    @property
    def rng(self):
        """The numpy random number generator."""
        return self._rng


class PairedSpectrumDataset(SpectrumDataset):
    """Parse and retrieve collections of mass spectra.

    Parameters
    ----------
    spectrum_index : depthcharge.data.SpectrumIndex
        The collection of spectra to use as a dataset.
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. ``None``
        retains all of the peaks.
    n_pairs : int, optional
        The number of randomly paired spectra to use for evaluation.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    tol : float, optional
        The tolerance with which to calculate the generalized scalar product.
    p_close : float, optional
        The probability of pairs occuring within a number of indices defined
        by ``close_scans``. The idea here is to try and enrich for pairs
        that have a higher similarity, such as those that occur within a few
        scans of each other.
    close_scans : int, optional
        If looking for a "close" pair of spectra, this is the number of scans
        within which to look.
    random_state : int or RandomState, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.

    Attributes
    ----------
    n_peaks : int
        The maximum number of mass speak to consider for each mass spectrum.
    n_pairs : int
        The number of pairs to generate.
    min_mz : float
        The minimum m/z to consider for each mass spectrum.
    tol : float
        The tolerance for calculating the GSP.
    p_close : float
        The probability of pairs occuring within a number of indices defined
        by ``close_scans``.
    close_scans : int
        If looking for a "close" pair of spectra, this is the number of scans
        within which to look.
    n_spectra : int
    index : depthcharge.data.SpectrumIndex
    rng : numpy.random.Generator
    """

    def __init__(
        self,
        spectrum_index,
        n_peaks=200,
        n_pairs=10000,
        min_mz=140,
        tol=0.01,
        p_close=0.0,
        close_scans=50,
        random_state=42,
    ):
        """Initialize a PairedSpectrumDataset."""
        super().__init__(
            spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            random_state=random_state,
        )

        self._eval = False
        self.n_pairs = n_pairs
        self.tol = tol
        self.p_close = p_close
        self.close_scans = close_scans
        self._pairs = self._generate_pairs(self.n_spectra)

    def __len__(self):
        """The number of pairs."""
        return self.n_pairs

    def __getitem__(self, idx):
        """Return a single pair of mass spectra.

        Parameters
        ----------
        idx : int
            The index to return.

        Returns
        -------
        spectrum_x : torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum_x[:, 0]`` are the m/z values and
            ``spectrum_y[:, 1]`` are their associated intensities.
        spectrum_y : torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum_y[:, 0]`` are the m/z values and
            ``spectrum_y[:, 1]`` are their associated intensities.
        gsp : float
            The calculated GSP between the two mass spectra.
        """
        indices = self._pairs[[idx], :]
        x_spec, _, _ = super().__getitem__(indices[0, 0])
        y_spec, _, _ = super().__getitem__(indices[0, 1])
        sim = similarity.gsp(np.array(x_spec), np.array(y_spec), self.tol)
        return x_spec, y_spec, sim

    def _generate_pairs(self, n_pairs=1):
        """Select a pair of spectra.

        Parameters
        ----------
        n_pairs : int, optional
            The number of pairs to generate.

        Returns
        -------
        numpy.ndarray of shape (n_pairs, 2)
            The indices of spectra to pair.
        """
        indices = self.rng.integers(self.n_spectra, size=(n_pairs, 2))
        replace = self.rng.binomial(1, self.p_close, size=n_pairs).astype(bool)
        new_idx = indices[:, 0] + self.rng.integers(
            low=-self.close_scans,
            high=self.close_scans,
            size=n_pairs,
        )
        indices[replace, 1] = new_idx[replace]
        return indices


class PairedSpectrumStreamer(IterableDataset, PairedSpectrumDataset):
    """Parse and retrieve collections of mass spectra.

    Parameters
    ----------
    spectrum_index : depthcharge.data.SpectrumIndex
        The collection of spectra to use as a dataset.
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. ``None``
        retains all of the peaks.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    tol : float, optional
        The tolerance with which to calculate the generalized scalar product.
    p_close : float, optional
        The probability of pairs occuring within a number of indices defined
        by ``close_scans``. The idea here is to try and enrich for pairs
        that have a higher similarity, such as those that occur within a few
        scans of each other.
    close_scans : int, optional
        If look for a "close" pair of spectra, this is the number of scans
        within which to look.
    random_state : int or RandomState, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.

    Attributes
    ----------
    n_peaks : int
        The maximum number of mass speak to consider for each mass spectrum.
    min_mz : float
        The minimum m/z to consider for each mass spectrum.
    tol : float
        The tolerance for calculating the GSP.
    p_close : float
        The probability of pairs occuring within a number of indices defined
        by ``close_scans``.
    close_scans : int
        If looking for a "close" pair of spectra, this is the number of scans
        within which to look.
    n_spectra : int
    index : depthcharge.data.SpectrumIndex
    rng : numpy.random.Generator
    """

    def __init__(
        self,
        spectrum_index,
        n_peaks=200,
        min_mz=140,
        tol=0.01,
        p_close=0.0,
        close_scans=50,
        random_state=42,
    ):
        """Initialize a PairedSpectrumDataset"""
        super().__init__(
            spectrum_index,
            n_peaks=n_peaks,
            n_pairs=1,
            min_mz=min_mz,
            tol=tol,
            p_close=p_close,
            close_scans=close_scans,
            random_state=random_state,
        )

    def __len__(self):
        """Undefined for an torch.utils.data.IterableDataset."""
        # See https://github.com/pytorch/pytorch/blob/ab5e1c69a7e77f79df64b6ad3b04cff72e09807b/torch/utils/data/sampler.py#L26-L51
        raise TypeError

    def __iter__(self):
        """Generate random pairs."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self._rng = np.random.default_rng(self.rng.integers(999999))

        while True:
            yield self[0]

    def __getitem__(self, idx):
        """Return a single pair of mass spectra.

        Parameters
        ----------
        idx : int
            The index to return.

        Returns
        -------
        spectrum_x : torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum_x[:, 0]`` are the m/z values and
            ``spectrum_y[:, 1]`` are their associated intensities.
        spectrum_y : torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum_y[:, 0]`` are the m/z values and
            ``spectrum_y[:, 1]`` are their associated intensities.
        gsp : float
            The calculated GSP between the two mass spectra.
        """
        indices = self._generate_pairs()
        x_spec, _, _ = super().__getitem__(indices[0, 0])
        y_spec, _, _ = super().__getitem__(indices[0, 1])
        sim = similarity.gsp(np.array(x_spec), np.array(y_spec), self.tol)
        return x_spec, y_spec, sim


class AnnotatedSpectrumDataset(SpectrumDataset):
    """Parse and retrieve collections of mass spectra

    Parameters
    ----------
    annotated_spectrum_index : depthcharge.data.SpectrumIndex
        The collection of annotated mass spectra to use as a dataset.
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. ``None``
        retains all of the peaks.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    random_state : int or RandomState, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.

    Attributes
    ----------
    n_peaks : int
        The maximum number of mass speak to consider for each mass spectrum.
    min_mz : float
        The minimum m/z to consider for each mass spectrum.
    n_spectra : int
    index : depthcharge.data.SpectrumIndex
    rng : numpy.random.Generator
    """

    def __init__(
        self,
        annotated_spectrum_index,
        n_peaks=200,
        min_mz=140,
        random_state=None,
    ):
        """Initialize an AnnotatedSpectrumDataset"""
        super().__init__(
            annotated_spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            random_state=random_state,
        )

    def __getitem__(self, idx):
        """Return a single annotated mass spectrum.

        Parameters
        ----------
        idx : int
            The index to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum[:, 0]`` are the m/z values and
            ``spectrum[:, 1]`` are their associated intensities.
        precursor_mz : float
            The m/z of the precursor.
        precursor_charge : int
            The charge of the precursor.
        annotation : str
            The annotation for the mass spectrum.
        """
        spec_idx = self._order[idx]
        mz_array, int_array, prec_mz, prec_charge, pep = self.index[spec_idx]
        spec = self._process_peaks(mz_array, int_array)
        return spec, prec_mz, prec_charge, pep
