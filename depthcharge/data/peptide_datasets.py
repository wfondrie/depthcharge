"""Datasets for working with peptide sequences."""
from collections.abc import Iterable

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader, Dataset


class PeptideDataset(Dataset):
    """A dataset for peptide sequences.

    Parameters
    ----------
    sequences : ArrayLike of str
        The peptide sequences in a format compatible with
        your tokenizer. ProForma is preferred.
    charges : ArrayLike of int
        The charge state for each peptide.
    target : torch.Tensor, optional
        Target values to predict.
    """

    def __init__(
        self,
        sequences: ArrayLike,
        charges: ArrayLike,
        target: ArrayLike | None = None,
    ) -> None:
        """Initialize a PeptideDataset."""
        self.sequences = sequences
        self.charges = charges
        if len(self.sequences) != len(self.charges):
            raise ValueError(
                "'sequences' and 'charges' must be the same length."
            )

        self.target = target
        if self.target is not None and len(self.target) != len(self.charges):
            raise ValueError(
                "'sequences' and 'target' must be the same length."
            )

    def __getitem__(self, idx: int) -> tuple[str, int, float | None]:
        """Get a single example."""
        seq, charge = self.sequences[idx], self.charges[idx]
        if self.target is not None:
            return seq, charge, self.target[idx]

        return seq, charge, None

    def __len__(self) -> int:
        """The number of peptides in the data."""
        return len(self.sequences)

    @staticmethod
    def collate_fn(
        batch: Iterable[tuple[str, int]],
    ) -> tuple[np.ndarray, torch.Tensor]:
        """The collate function for a PeptideDataset.

        The peptides are paired with their charge states. Unfortunately,
        there is no support for strings in PyTorch currently, which is why
        this is needed at all.

        Parameters
        ----------
        batch : Iterable[tuple[str, int]]
            The peptide sequehnce-charge state pairs.

        Returns
        -------
        sequences : np.ndarray[str]
            The peptide sequences.
        charges : torch.Tensor[int]
            The charge states.
        """
        sequences, charges, target = list(zip(*batch))
        sequences = np.array(sequences)
        charges = torch.tensor(charges, dtype=int)
        if any(t is None for t in target):
            target = None
        else:
            target = torch.tensor(target)

        return sequences, charges, target

    def loader(self, *args: tuple, **kwargs: dict) -> DataLoader:
        """A PyTorch DataLoader for peptides.

        Parameters
        ----------
        *args : tuple
            Arguments passed initialize a torch.utils.data.DataLoader,
            excluding ``dataset`` and ``collate_fn``.
        **kwargs : dict
            Keyword arguments passed initialize a torch.utils.data.DataLoader,
            excluding ``dataset`` and ``collate_fn``.

        Returns
        -------
        torch.utils.data.DataLoader
            A DataLoader for the peptide.
        """
        return DataLoader(self, *args, collate_fn=self.collate_fn, **kwargs)
