"""A PyTorch Dataset class for annotated spectra."""
import torch
import numpy as np
from torch.utils.data import Dataset

from . import preprocessing
from .. import utils


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

    Attributes
    ----------
    n_peaks : int
        The maximum number of mass speak to consider for each mass spectrum.
    min_mz : float
        The minimum m/z to consider for each mass spectrum.
    n_spectra : int
    index : depthcharge.data.SpectrumIndex
    """

    def __init__(
        self,
        spectrum_index,
        n_peaks=200,
        min_mz=140,
        preprocessing_fn=None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.preprocessing_fn = preprocessing_fn
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
        spec = self._process_peaks(*batch)
        if not spec.sum():
            spec = torch.tensor([[0, 1]]).float()

        return spec, batch[2], batch[3]

    def get_spectrum_id(self, idx):
        """Return the identifier for a single mass spectrum.

        Parameters
        ----------
        idx : int
            The index of a mass spectrum within the SpectrumIndex.

        Returns
        -------
        ms_data_file : str
            The mass spectrometry data file from which the mass spectrum was
            originally parsed.
        identifier : str
            The mass spectrum identifier, per PSI recommendations.
        """
        with self.index:
            return self.index.get_spectrum_id(idx)

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

    Attributes
    ----------
    n_peaks : int
        The maximum number of mass speak to consider for each mass spectrum.
    min_mz : float
        The minimum m/z to consider for each mass spectrum.
    n_spectra : int
    index : depthcharge.data.AnnotatedSpectrumIndex
    """

    def __init__(
        self,
        annotated_spectrum_index,
        n_peaks=200,
        min_mz=140,
        preprocessing_fn=None,
    ):
        """Initialize an AnnotatedSpectrumDataset"""
        super().__init__(
            annotated_spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            preprocessing_fn=preprocessing_fn,
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
        *batch, seq = self.index[idx]
        spec = self._process_peaks(*batch)
        if not spec.sum():
            spec = torch.tensor([[0, 1]]).float()

        return spec, batch[2], batch[3], seq

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
        spec, mz, charge, seq = list(zip(*batch))
        charge = torch.tensor(charge)
        mass = (torch.tensor(mz) - 1.007276) * charge
        precursors = torch.vstack([mass, charge]).T.float()
        spec = torch.nn.utils.rnn.pad_sequence(spec, batch_first=True)
        return spec, precursors, np.array(seq)
