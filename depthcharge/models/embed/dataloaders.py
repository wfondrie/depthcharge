"""Data loaders for the embedding task.

This module also ensure consistent train, validation, and test splits.
"""
import os
import json
from pathlib import Path
from functools import partial

import ppx
import torch
import numpy as np
import pytorch_lightning as pl

from ...data import (
    PairedSpectrumDataset,
    PairedSpectrumStreamer,
    SpectrumIndex,
)

SPLITS = Path(__file__).parent / "SPLITS.json"


class PairedSpectrumDataModule(pl.LightningDataModule):
    """Prepare data for a PairedSpectrumEncoder.

    Parameters
    ----------
    train_index_path : str or Path
        The path to the HDF5 spectrum index file for training. If it does not
        exist, a new one will be created.
    valid_index_path : str or Path
        The path to the HDF5 spectrum index file for validation. If it does not
        exist, a new one will be created.
    test_index_path : str or Path
        The path to the HDF5 spectrum index file for testing. If it does not
        exist, a new one will be created.
    splits_path : str or Path, optional
        The path to a JSON file describing which MassIVE projects should be
        associated with each split. If ``None``, splits included with
        depthcharge will be used.
    batch_size : int, optional
        The batch size to use for training and evaluations
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
        exclude TMT and iTRAQ reporter ions.
    p_close : float, optional
        The probability of pairs occuring within a number of indices defined
        by ``close_scans``. The idea here is to try and enrich for pairs
        that have a higher similarity, such as those that occur within a few
        scans of each other.
    close_scans : int, optional
        If look for a "close" pair of spectra, this is the number of scans
        within which to look.
    num_workers : int, optional
        The number of workers to use for data loading. By default, the number
        of available CPU cores on the current machine is used.
    keep_files : bool, optional
        Keep the downloaded mzML files? Otherwise, the downloaded files are
        deleted after they have been added to a spectrum index.
    random_state : int or Generator, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.
    """

    def __init__(
        self,
        train_index_path,
        valid_index_path,
        test_index_path,
        splits_path=None,
        batch_size=128,
        n_peaks=200,
        n_pairs=10000,
        min_mz=140,
        tol=0.01,
        p_close=0.0,
        close_scans=50,
        num_workers=None,
        keep_files=False,
        random_state=None,
    ):
        """Initialize the PairedSpectrumDataModule."""
        super().__init__()
        self.train_index = SpectrumIndex(train_index_path)
        self.valid_index = SpectrumIndex(valid_index_path)
        self.test_index = SpectrumIndex(test_index_path)
        self.batch_size = batch_size
        self.n_peaks = n_peaks
        self.n_pairs = n_pairs
        self.min_mz = min_mz
        self.tol = tol
        self.p_close = p_close
        self.close_scans = close_scans
        self.num_workers = num_workers
        self.keep_files = keep_files
        self.rng = np.random.default_rng(random_state)
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        if self.num_workers is None:
            self.num_workers = os.cpu_count()

        if splits_path is None:
            splits_path = SPLITS
        else:
            splits_path = Path(splits_path)

        with splits_path.open("r") as split_data:
            self.splits = json.load(split_data)

    def prepare_data(self):
        """Verify that the spectrum indices contain the expected spectra.

        If a mass spectrometry data file is missing from a spectrum index,
        this function will download the missing mzML file and add it.
        """
        index_map = {
            "train": self.train_index,
            "validation": self.valid_index,
            "test": self.test_index,
        }

        for split, projects in self.splits.items():
            if split == "rejected":
                continue

            index = index_map[split]
            for project, ms_files in projects.items():
                for ms_file in ms_files:
                    fname = str(Path(project, ms_file))
                    if any(fname in f for f in index.ms_files):
                        continue

                    proj = ppx.find_project(project)
                    downloaded = proj.download(ms_file)[0]
                    index.add_file(downloaded)
                    if not self.keep_files:
                        downloaded.unlink()  # Delete the file when we're done

    def setup(self, stage=None):
        """Set up the PyTorch Datasets.

        Parameters
        ----------
        stage : str {"fit", "validate", "test"}
            The stage indicating which Datasets to prepare. All are prepared
            by default.
        """
        make_dataset = partial(
            PairedSpectrumDataset,
            n_peaks=self.n_peaks,
            n_pairs=self.n_pairs,
            min_mz=self.min_mz,
            tol=self.tol,
            p_close=self.p_close,
            close_scans=self.close_scans,
            random_state=self.rng,
        )

        if stage in (None, "fit", "validate"):
            self.train_dataset = PairedSpectrumStreamer(
                self.train_index,
                n_peaks=self.n_peaks,
                min_mz=self.min_mz,
                tol=self.tol,
                p_close=self.p_close,
                close_scans=self.close_scans,
                random_state=self.rng,
            )
            self.valid_dataset = make_dataset(self.valid_index)

        if stage in (None, "test"):
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
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=prepare_batch,
            pin_memory=True,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        """Get the validation DataLoader."""
        return self._make_loader(self.valid_dataset)

    def test_dataloader(self):
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset)


def prepare_batch(batch):
    """This is the collate function.

    The mass spectra must be padded so that they fit nicely as a tensor.
    However, the padded elements are ignored during the subsequent steps.

    Parameters
    ----------
    batch : tuple of tuple of torch.tensor
        A batch of data from a PairedSpectrumDataset.

    Returns
    -------
    X : torch.Tensor of shape (batch_size, n_peaks, 2)
    Y : torch.Tensor of shape (batch_size, n_peaks, 2)
        The mass spectra to embed, where ``X[:, :, 0]`` are the m/z values
        and ``X[:, :, 1]`` are their associated intensities.
    gsp : torch.Tensor of shape (batch_size,)
        The GSP calculated between the mass spectra in X and Y.
    """
    X, Y, gsp = list(zip(*batch))
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True)
    gsp = torch.tensor(gsp)
    return X, Y, gsp


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.rng = torch.randint(1, 99999, (1,)).item()
