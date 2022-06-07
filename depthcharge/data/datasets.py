"""A PyTorch Dataset class for annotated spectra."""
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset

from . import preprocessing
from .. import utils
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
    preprocessing_fn: Callable or list of Callable, optional
        The function(s) used to preprocess the mass spectra. See the
        preprocessing module for details. ``None``, the default, square root
        transforms the intensities and scales them to unit norm.
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
        preprocessing_fn=None,
        random_state=None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.preprocessing_fn = preprocessing_fn
        self.rng = np.random.default_rng(random_state)
        self._index = spectrum_index

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
        batch = self.index[idx]
        spec, mz, charge, seq, scan_id = batch
        spec = self._process_peaks(spec, mz, charge, seq)
        if not spec.sum():
            spec = torch.tensor([[0, 1]]).float()

        return spec, batch[2], batch[3]

    def _process_peaks(
        self,
        mz_array,
        int_array,
        precursor_mz,
        precursor_charge,
    ):
        """Process each mass spectrum.

        Parameters
        ----------
        mz_array : numpy.ndarray of shape (n_peaks,)
            The m/z values of the peaks in the spectrum.
        int_array : numpy.ndarray of shape (n_peaks,)
            The intensity values of the peaks in the spectrum.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.

        Returns
        -------
        torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where column 0 are the m/z values and
            column 1 are their associated intensities.
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

        mz_array = torch.tensor(mz_array)
        int_array = torch.tensor(int_array)
        for func in self.preprocessing_fn:
            mz_array, int_array = func(
                mz_array=mz_array,
                int_array=int_array,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
            )

        return torch.vstack([mz_array, int_array]).T.float()

    @property
    def n_spectra(self):
        """The total number of spectra."""
        return self.index.n_spectra

    @property
    def index(self):
        """The underyling SpectrumIndex."""
        return self._index

    @property
    def preprocessing_fn(self):
        """The functions to preprocess each mass spectrum."""
        return self._preprocessing_fn

    @preprocessing_fn.setter
    def preprocessing_fn(self, funcs):
        """Set the preprocessing_fn"""
        if funcs is None:
            funcs = preprocessing.sqrt_and_norm

        self._preprocessing_fn = utils.listify(funcs)

    @property
    def rng(self):
        """The numpy random number generator."""
        return self._rng

    @rng.setter
    def rng(self, seed):
        """Set the numpy random number generator."""
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def collate_fn(batch):
        """This is the collate function for a SpectrumDataset.

        The mass spectra must be padded so that they fit nicely as a tensor.
        However, the padded elements are ignored during the subsequent steps.

        Parameters
        ----------
        batch : tuple of tuple of torch.Tensor
            A batch of data from an AnnotatedSpectrumDataset.

        Returns
        -------
        spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
            The mass spectra to sequence, where ``X[:, :, 0]`` are the m/z
            values and ``X[:, :, 1]`` are their associated intensities.
        precursors : torch.Tensor of shape (batch_size, 2)
            The precursor mass and charge state.
        """
        spec, mz, charge = list(zip(*batch))
        charge = torch.tensor(charge)
        mass = (torch.tensor(mz) - 1.007276) * charge
        precursors = torch.vstack([mass, charge]).T.float()
        spec = torch.nn.utils.rnn.pad_sequence(
            spec,
            batch_first=True,
        )
        return spec, precursors


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
        self._pairs = self._generate_pairs(self.n_pairs)

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
        indices = self.rng.integers(self.n_spectra - 1, size=(n_pairs, 2))
        replace = self.rng.binomial(1, self.p_close, size=n_pairs).astype(bool)
        new_idx = indices[:, 0] + self.rng.integers(
            low=-self.close_scans,
            high=self.close_scans,
            size=n_pairs,
        )

        too_big = np.nonzero(new_idx >= self.n_spectra)
        new_idx[too_big] = new_idx[too_big] - self.n_spectra
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
            num = worker_info.id
        else:
            num = 0

        while True:
            indices = self._generate_pairs(num + 1)[num, :]
            x_spec, _, _ = SpectrumDataset.__getitem__(self, indices[0])
            y_spec, _, _ = SpectrumDataset.__getitem__(self, indices[1])
            sim = similarity.gsp(np.array(x_spec), np.array(y_spec), self.tol)
            yield x_spec, y_spec, sim

    def __getitem__(self, idx):
        """Return a single pair of mass spectra.

        Parameters
        ----------
        idx : int
            This is essentially meaningless.

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
        return torch.utils.data.IterableDataset.__getitem__(self, idx)


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
    preprocessing_fn: Callable or list of Callable, optional
        The function(s) used to preprocess the mass spectra. See the
        preprocessing module for details. ``None``, the default, square root
        transforms the intensities and scales them to unit norm.
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
        preprocessing_fn=None,
        random_state=None,
    ):
        """Initialize an AnnotatedSpectrumDataset"""
        super().__init__(
            annotated_spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            preprocessing_fn=preprocessing_fn,
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
        *batch, pep, scan_id = self.index[idx]
        spec = self._process_peaks(*batch)
        return spec, batch[2], batch[3], pep, scan_id

    @staticmethod
    def collate_fn(batch):
        """This is the collate function for an AnnotatedSpectrumDataset.

        The mass spectra must be padded so that they fit nicely as a tensor.
        However, the padded elements are ignored during the subsequent steps.

        Parameters
        ----------
        batch : tuple of tuple of torch.Tensor
            A batch of data from an AnnotatedSpectrumDataset.

        Returns
        -------
        spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
            The mass spectra to sequence, where ``X[:, :, 0]`` are the m/z
            values and ``X[:, :, 1]`` are their associated intensities.
        precursors : torch.Tensor of shape (batch_size, 2)
            The precursor mass and charge state.
        sequence : list of str
            The peptide sequence annotations.

        """
        spec, mz, charge, seq, scan_id = list(zip(*batch))
        charge = torch.tensor(charge)
        mass = (torch.tensor(mz) - 1.007276) * charge
        precursors = torch.vstack([mass, charge]).T.float()
        spec = torch.nn.utils.rnn.pad_sequence(spec, batch_first=True)
        return spec, precursors, np.array(seq), scan_id

