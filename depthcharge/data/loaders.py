"""pytorch-lightning LightningDataModules for various tasks."""
import os

import torch
import numpy as np
from pytorch_lightning import LightningDataModule

from .hdf5 import SpectrumIndex, AnnotatedSpectrumIndex
from .datasets import SpectrumDataset, AnnotatedSpectrumDataset


class SpectrumDataModule(LightningDataModule):
    """Data loaders for mass spectrometry data.

    Parameters
    ----------
    train_index : SpectrumIndex or AnnotatedSpectrumIndex
        The spectrum index file for training.
    val_index : SpectrumIndex or AnnotatedSpectrumIndex
        The spectrum index file for validation.
    test_index : SpectrumIndex or AnnotatedSpectrumIndex
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
    shuffle : Boolean
        Whether the batches are shuffled or not for training. The batches are
        not shuffled for validation or testing.
    """

    def __init__(
        self,
        train_index=None,
        val_index=None,
        test_index=None,
        batch_size=128,
        n_peaks=200,
        min_mz=140,
        preprocessing_fn=None,
        num_workers=None,
        shuffle=True,
    ):
        """Initialize the PairedSpectrumDataModule."""
        super().__init__()
        self.train_index = train_index
        self.val_index = val_index
        self.test_index = test_index
        self.batch_size = batch_size
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.preprocessing_fn = preprocessing_fn
        self.num_workers = num_workers
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.shuffle = shuffle

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
        if stage in (None, "fit", "validate"):
            if self.train_index is not None:
                self.train_dataset = self._make_dataset(self.train_index)

            if self.val_index is not None:
                self.val_dataset = self._make_dataset(self.val_index)

        if stage in (None, "test"):
            if self.test_index is not None:
                self.test_dataset = self._make_dataset(self.test_index)

    def _make_dataset(self, index, random_state=None):
        """Create a PyTorch Dataset.

        Parameters
        ----------
        index : SpectrumIndex or AnnotatedSpectrumIndex

        Returns
        -------
        SpectrumDataset or AnnotatedSpectrumDataset
            The instantiated dataset.
        """
        if isinstance(index, AnnotatedSpectrumIndex):
            dataset_class = AnnotatedSpectrumDataset
        elif isinstance(index, SpectrumIndex):
            dataset_class = SpectrumDataset
        else:
            raise ValueError(
                "index must be a SpectrumIndex or AnnotatedSpectrumIndex."
            )

        return dataset_class(
            index,
            n_peaks=self.n_peaks,
            min_mz=self.min_mz,
            preprocessing_fn=self.preprocessing_fn,
        )

    def _make_loader(self, dataset, shuffle):
        """Create a PyTorch DataLoader.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            A PyTorch Dataset.
        shuffle : bool
            Shuffle the batches?

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader.
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        """Get the training DataLoader."""
        return self._make_loader(self.train_dataset, self.shuffle)

    def val_dataloader(self):
        """Get the validation DataLoader."""
        return self._make_loader(self.val_dataset, False)

    def test_dataloader(self):
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset, False)
