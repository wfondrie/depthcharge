"""Datasets for working with peptide sequences."""

from collections.abc import Iterable

import torch
from torch.utils.data import TensorDataset

from ..tokenizers import Tokenizer


class AnalyteDataset(TensorDataset):
    """A dataset for peptide sequences.

    Parameters
    ----------
    tokenizer : PeptideTokenizer
        A tokenizer specifying how to transform peptide sequences.
        into tokens.
    sequences : Iterable[str]
        The peptide sequences in a format compatible with
        your tokenizer. ProForma is preferred.
    *args : torch.Tensor, optional
        Additional values to include during data loading.

    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        sequences: Iterable[str],
        *args: torch.Tensor,
    ) -> None:
        """Initialize a PeptideDataset."""
        tokens = tokenizer.tokenize(sequences)
        super().__init__(tokens, *args)

    @property
    def tokens(self) -> torch.Tensor:
        """The peptide sequence tokens."""
        return self.tensors[0]
