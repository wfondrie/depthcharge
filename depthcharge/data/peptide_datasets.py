"""Datasets for working with peptide sequences."""
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..tokenizers import PeptideTokenizer


class PeptideDataset(TensorDataset):
    """A dataset for peptide sequences.

    Parameters
    ----------
    tokenizer : PeptideTokenizer
        A tokenizer specifying how to transform peptide sequences.
        into tokens.
    sequences : Iterable[str]
        The peptide sequences in a format compatible with
        your tokenizer. ProForma is preferred.
    charges : torch.Tensor,
        The charge state for each peptide.
    *args : torch.Tensor, optional
        Additional values to include during data loading.
    """

    def __init__(
        self,
        tokenizer: PeptideTokenizer,
        sequences: Iterable[str],
        charges: torch.Tensor,
        *args: torch.Tensor,
    ) -> None:
        """Initialize a PeptideDataset."""
        tokens = tokenizer.tokenize(sequences)
        super().__init__(tokens, charges, *args)

    @property
    def tokens(self) -> torch.Tensor:
        """The peptide sequence tokens."""
        return self.tensors[0]

    @property
    def charges(self) -> torch.Tensor:
        """The peptide charges."""
        return self.tensors[1]

    def loader(self, *args: tuple, **kwargs: dict) -> DataLoader:
        """A PyTorch DataLoader for peptides.

        Parameters
        ----------
        *args : tuple
            Arguments passed initialize a torch.utils.data.DataLoader,
            excluding ``dataset``.
        **kwargs : dict
            Keyword arguments passed initialize a torch.utils.data.DataLoader,
            excluding ``dataset``.

        Returns
        -------
        torch.utils.data.DataLoader
            A DataLoader for the peptide.
        """
        return DataLoader(self, *args, **kwargs)
