"""A base Tokenizer class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import torch
from sortedcontainers import SortedDict
from torch import nn


class Tokenizer(ABC):
    """An abstract base class for Depthcharge tokenizers.

    Parameters
    ----------
    tokens : Sequence[str]
        The tokens to consider.
    stop_token : str, optional
        The stop token to use.
    """

    def __init__(self, tokens: Sequence[str], stop_token: str = "$") -> None:
        """Initialize a tokenizer."""
        self.stop_token = stop_token
        self.index = SortedDict({k: i for i, k in enumerate(tokens)})
        if self.stop_token in self.index.keys():
            raise ValueError(
                f"Stop token {stop_token} already exists in tokens.",
            )

        self.index[self.stop_token] = len(self.index)
        self.reverse_index = [None] + list(self.index.keys)

    @abstractmethod
    def split(self, sequence: str) -> list[str]:
        """Split a sequence into the constituent string tokens."""

    def tokenize(
        self,
        sequences: Iterable[str],
        to_strings: bool = False,
    ) -> torch.Tensor | list[list[str]]:
        """Tokenize the input sequences.

        Parameters
        ----------
        sequences : Iterable[str]
            The sequences to tokenize.
        to_strings : bool, optional
            Return each as a list of token strings rather than a
            tensor. This is useful for debugging.

        Returns
        -------
        torch.Tensor of shape (n_sequences, max_length) or list[list[str]]
            Either a tensor containing the integer values for each
            token, padded with 0's, or the list of tokens comprising
            each sequence.
        """
        try:
            if to_strings:
                return [self.split(s) for s in sequences]

            tokens = [self.index[t] for s in sequences for t in self.split(s)]
            return nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        except KeyError as err:
            raise ValueError("Unrecognized token") from err

    def detokenize(
        self,
        tokens: torch.Tensor,
        join: bool = True,
        trim_stop_token: bool = True,
    ) -> list[str] | list[list[str]]:
        """Retreive sequences from tokens.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_sequences, max_length)
            The zero-padded tensor of integerized tokens to decode.
        join : bool, optional
            Join tokens into strings?
        trim_stop_token : bool, optional
            Remove the stop token from the end of a sequence.

        Returns
        -------
        list[str] or list[list[str]]
            The decoded sequences each as a string or list or strings.
        """
        decoded = []
        for row in tokens:
            seq = [
                self.reverse_index[i]
                for i in row
                if self.reverse_index[i] is not None
            ]

            if trim_stop_token and seq[-1] == self.stop_token:
                seq.pop(-1)

            if join:
                seq = "".join(seq)

            decoded.append(seq)

        return decoded
