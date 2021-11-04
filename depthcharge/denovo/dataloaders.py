"""Data loaders for the de novo sequencing task.

This module also ensure consistent train, validation, and test splits.
"""
import os
from functools import partial

import torch
import numpy as np
import pytorch_lightning as pl

from ..data import AnnotatedSpectrumDataset


class DeNovoDataModule(pl.LightningDataModule):
    """Prepare data for a SiameseSpectrumEncoder.

    Parameters
    ----------
    train_index : AnnotatedSpectrumIndex
        The spectrum index file for training.
    valid_index : AnnotatedSpectrumIndex
        The spectrum index file for validation.
    test_index : AnnotatedSpectrumIndex
        The spectrum index file for testing.
    batch_size : int, optional
        The batch size to use for training and evaluations
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. ``None``
        retains all of the peaks.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    num_workers : int, optional
        The number of workers to use for data loading. By default, the number
        of available CPU cores on the current machine is used.
    random_state : int or Generator, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.
    """

    def __init__(
        self,
        train_index=None,
        valid_index=None,
        test_index=None,
        batch_size=128,
        n_peaks=200,
        min_mz=140,
        num_workers=None,
        random_state=None,
    ):
        """Initialize the PairedSpectrumDataModule."""
        super().__init__()
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        self.batch_size = batch_size
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.num_workers = num_workers
        self.rng = np.random.default_rng(random_state)
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        if self.num_workers is None:
            self.num_workers = os.cpu_count()

    def setup(self, stage=None):
        """Set up the PyTorch Datasets.

        Parameters
        ----------
        stage : str {"fit", "validate", "test"}
            The stage indicating which Datasets to prepare. All are prepared
            by default.
        """
        make_dataset = partial(
            AnnotatedSpectrumDataset,
            n_peaks=self.n_peaks,
            min_mz=self.min_mz,
        )

        if stage in (None, "fit", "validate"):
            if self.train_index is not None:
                self.train_dataset = make_dataset(
                    self.train_index,
                    random_state=self.rng,
                )

            if self.valid_index is not None:
                self.valid_dataset = make_dataset(self.valid_index)

        if stage in (None, "test"):
            if self.test_index is not None:
                self.test_dataset = make_dataset(self.test_index)

    def _make_loader(self, dataset):
        """Create a PyTorch DataLoader.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            A PyTorch Dataset.

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader.
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=prepare_batch,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        """Get the training DataLoader."""
        return self._make_loader(self.train_dataset)

    def val_dataloader(self):
        """Get the validation DataLoader."""
        return self._make_loader(self.valid_dataset)

    def test_dataloader(self):
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset)


def prepare_batch(batch):
    """This is the collate function

    The mass spectra must be padded so that they fit nicely as a tensor.
    However, the padded elements are ignored during the subsequent steps.

    Parameters
    ----------
    batch : tuple of tuple of torch.Tensor
        A batch of data from an AnnotatedSpectrumDataset.

    Returns
    -------
    spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
        The mass spectra to sequence, where ``X[:, :, 0]`` are the m/z values
        and ``X[:, :, 1]`` are their associated intensities.
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
    return spec, precursors, seq
