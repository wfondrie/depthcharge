"""A base Tokenizer class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import torch
from sortedcontainers import SortedDict, SortedSet
from torch import nn

from .. import utils


class Tokenizer(ABC):
    """An abstract base class for Depthcharge tokenizers.

    Parameters
    ----------
    tokens : Sequence[str]
        The tokens to consider.
    start_token : str, optional
        The start token to use.
    stop_token : str, optional
        The stop token to use.

    """

    def __init__(
        self,
        tokens: Sequence[str],
        start_token: str | None = None,
        stop_token: str | None = "$",
    ) -> None:
        """Initialize a tokenizer."""
        self.start_token = start_token
        self.stop_token = stop_token

        tokens = SortedSet(tokens)
        if self.stop_token in tokens:
            raise ValueError(
                f"Stop token {stop_token} already exists in tokens.",
            )

        if start_token is not None:
            tokens.add(self.start_token)
        if stop_token is not None:
            tokens.add(self.stop_token)

        self.index = SortedDict({k: i + 1 for i, k in enumerate(tokens)})
        self.reverse_index = [None] + list(tokens)  # 0 is padding.
        self.start_int = self.index.get(self.start_token, None)
        self.stop_int = self.index.get(self.stop_token, None)
        self.padding_int = 0

    def __len__(self) -> int:
        """The number of tokens."""
        return len(self.index)

    @abstractmethod
    def split(self, sequence: str) -> list[str]:
        """Split a sequence into the constituent string tokens."""

    def tokenize(
        self,
        sequences: Iterable[str] | str,
        add_start: bool = False,
        add_stop: bool = False,
        to_strings: bool = False,
    ) -> torch.tensor | list[list[str]]:
        """Tokenize the input sequences.

        Parameters
        ----------
        sequences : Iterable[str] or str
            The sequences to tokenize.
        add_start : bool, optional
            Prepend the start token to the beginning of the sequence.
        add_stop : bool, optional
            Append the stop token to the end of the sequence.
        to_strings : bool, optional
            Return each as a list of token strings rather than a
            tensor. This is useful for debugging.

        Returns
        -------
        torch.tensor of shape (n_sequences, max_length) or list[list[str]]
            Either a tensor containing the integer values for each
            token, padded with 0's, or the list of tokens comprising
            each sequence.

        """
        add_start = add_start and self.start_token is not None
        add_stop = add_stop and self.stop_token is not None
        try:
            out = []
            for seq in utils.listify(sequences):
                tokens = self.split(seq)
                if add_start and tokens[0] != self.start_token:
                    tokens.insert(0, self.start_token)

                if add_stop and tokens[-1] != self.stop_token:
                    tokens.append(self.stop_token)

                if to_strings:
                    out.append(tokens)
                    continue

                out.append(torch.tensor([self.index[t] for t in tokens]))

            if to_strings:
                return out

            return nn.utils.rnn.pad_sequence(out, batch_first=True)
        except KeyError as err:
            raise ValueError("Unrecognized token") from err

    def detokenize(
        self,
        tokens: torch.Tensor,
        join: bool = True,
        trim_start_token: bool = True,
        trim_stop_token: bool = True,
    ) -> list[str] | list[list[str]]:
        """Retreive sequences from tokens.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_sequences, max_length)
            The zero-padded tensor of integerized tokens to decode.
        join : bool, optional
            Join tokens into strings?
        trim_start_token : bool, optional
            Remove the start token from the beginning of a sequence.
        trim_stop_token : bool, optional
            Remove the stop token and anything following it from the sequence.

        Returns
        -------
        list[str] or list[list[str]]
            The decoded sequences each as a string or list or strings.

        """
        decoded = []
        for row in tokens:
            seq = []
            for idx in row:
                if self.reverse_index[idx] is None:
                    continue

                if trim_stop_token and idx == self.stop_int:
                    break

                seq.append(self.reverse_index[idx])

            if trim_start_token and seq[0] == self.start_token:
                seq.pop(0)

            if join:
                seq = "".join(seq)

            decoded.append(seq)

        return decoded
