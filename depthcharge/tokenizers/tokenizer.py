"""A base Tokenizer class."""

from __future__ import annotations

import copy
import heapq
import json
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from os import PathLike
from pathlib import Path

import torch
from sortedcontainers import SortedDict, SortedSet

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
    merges : Iterable[tuple[str, str]], optional
        Byte-pair encoding (BPE) merges, in the order that they are
        applied. Each merge joins two adjacent tokens into a new token.
        Merges are usually learned with `train_bpe()`.

    Attributes
    ----------
    merges : list[tuple[str, str]]
        The BPE merges, in the order that they are applied.
    expansions : list[tuple[str, ...]]
        The tokens returned by `split()` that comprise each token, where
        the list index is the integer representation for a token.
    bpe_dropout : float
        The probability of skipping each BPE merge during tokenization.
        This is a regularization technique for training (BPE-dropout).
        Set it to 0 for deterministic tokenization.

    """

    def __init__(
        self,
        tokens: Sequence[str],
        start_token: str | None = None,
        stop_token: str | None = "$",
        merges: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        """Initialize a tokenizer."""
        self.start_token = start_token
        self.stop_token = stop_token
        self.bpe_dropout = 0.0

        tokens = SortedSet(tokens)
        if self.stop_token in tokens:
            raise ValueError(
                f"Stop token {stop_token} already exists in tokens.",
            )

        if start_token is not None:
            tokens.add(self.start_token)
        if stop_token is not None:
            tokens.add(self.stop_token)

        self._base_tokens = list(tokens)
        self.padding_int = 0
        self._set_merges(merges)

    def __len__(self) -> int:
        """The number of tokens."""
        return len(self.index)

    def _set_merges(self, merges: Iterable[tuple[str, str]] | None) -> None:
        """Build the vocabulary from the base tokens and BPE merges.

        The base tokens are always assigned the same integers, and each
        token created by a merge is assigned the next integer.

        Parameters
        ----------
        merges : Iterable[tuple[str, str]], optional
            The BPE merges, in the order that they are applied.

        """
        self.index = SortedDict(
            {k: i + 1 for i, k in enumerate(self._base_tokens)}
        )
        self.reverse_index = [None] + list(self._base_tokens)  # 0 is padding.
        self.expansions = [()] + [(t,) for t in self._base_tokens]
        self.start_int = self.index.get(self.start_token, None)
        self.stop_int = self.index.get(self.stop_token, None)
        self.merges = []
        self._merge_ranks = {}
        self._merge_cache = {}

        special = {self.start_token, self.stop_token} - {None}
        for left, right in merges or []:
            pair = (left, right)
            if pair in self._merge_ranks:
                continue

            for token in pair:
                if token not in self.index:
                    raise ValueError(f"Unrecognized token in merge: {token}")
                if token in special:
                    raise ValueError(
                        f"Merges cannot include the special token {token}."
                    )

            expansion = (
                self.expansions[self.index[left]]
                + self.expansions[self.index[right]]
            )
            merged = left + right
            if merged in self.index:
                # Different merges can create the same token.
                if self.expansions[self.index[merged]] != expansion:
                    raise ValueError(
                        f"Merged token {merged} conflicts with an existing "
                        "token."
                    )
            else:
                self.index[merged] = len(self.reverse_index)
                self.reverse_index.append(merged)
                self.expansions.append(expansion)

            self._merge_ranks[pair] = len(self.merges)
            self.merges.append(pair)

    def _with_merges(self, merges: Iterable[tuple[str, str]]) -> Tokenizer:
        """Create a copy of the tokenizer with different BPE merges.

        Parameters
        ----------
        merges : Iterable[tuple[str, str]]
            The BPE merges, in the order that they are applied.

        Returns
        -------
        Tokenizer
            A new tokenizer with the BPE merges.

        """
        new = copy.deepcopy(self)
        new._set_merges(merges)
        return new

    def _apply_merges(
        self,
        tokens: list[str],
        dropout: float = 0.0,
    ) -> list[str]:
        """Apply the BPE merges to the tokens of a sequence.

        Parameters
        ----------
        tokens : list[str]
            The tokens returned by `split()`.
        dropout : float, optional
            The probability of skipping each merge.

        Returns
        -------
        list[str]
            The tokens after the merges are applied.

        """
        if not self._merge_ranks or len(tokens) < 2:
            return tokens

        if not dropout:
            key = tuple(tokens)
            cached = self._merge_cache.get(key)
            if cached is not None:
                return list(cached)

        out = _merge_tokens(tokens, self._merge_ranks, dropout)
        if not dropout:
            if len(self._merge_cache) >= 2**18:
                self._merge_cache.clear()

            self._merge_cache[key] = tuple(out)

        return out

    def train_bpe(
        self,
        sequences: Iterable[str] | str,
        vocab_size: int,
        min_frequency: int = 2,
        max_token_length: int | None = None,
    ) -> Tokenizer:
        """Learn byte-pair encoding (BPE) merges from sequences.

        BPE repeatedly merges the most frequent pair of adjacent tokens
        into a new token, similar to SentencePiece. Merges are learned on
        the tokens returned by `split()`, so a merge never divides a
        token such as a modified residue. Merges that the tokenizer
        already has are kept, and new merges are learned after them.

        Parameters
        ----------
        sequences : Iterable[str] or str
            The sequences from which to learn the merges.
        vocab_size : int
            The number of tokens in the new tokenizer, including the
            start and stop tokens. Training stops early if no pair of
            tokens is frequent enough.
        min_frequency : int, optional
            The minimum number of times that a pair of tokens must occur
            to be merged.
        max_token_length : int, optional
            The maximum number of tokens returned by `split()` that a
            merged token may contain.

        Returns
        -------
        Tokenizer
            A new tokenizer with the learned merges. This tokenizer is
            not changed.

        Examples
        --------
        >>> tokenizer = PeptideTokenizer().train_bpe(peptides, 100)
        >>> tokenizer.save_merges("merges.json")

        """
        counts = Counter()
        for seq in utils.listify(sequences):
            tokens = self._apply_merges(self.split(seq))
            for token in tokens:
                if token not in self.index:
                    raise ValueError(f"Unrecognized token: {token}")

            counts[tuple(tokens)] += 1

        vocab = dict(zip(self.reverse_index[1:], self.expansions[1:]))
        pairs = _PairCounter(
            counts=counts,
            lengths={t: len(e) for t, e in vocab.items()},
            special={self.start_token, self.stop_token},
            max_token_length=max_token_length,
        )

        merges = list(self.merges)
        while len(vocab) < vocab_size:
            pair = pairs.pop_best(min_frequency)
            if pair is None:
                break

            expansion = vocab[pair[0]] + vocab[pair[1]]
            if vocab.setdefault(pair[0] + pair[1], expansion) != expansion:
                pairs.ban(pair)  # Conflicts with an existing token.
                continue

            pairs.merge(pair)
            merges.append(pair)

        return self._with_merges(merges)

    def save_merges(self, path: str | PathLike) -> None:
        """Save the BPE merges to a JSON file.

        Parameters
        ----------
        path : str or PathLike
            The JSON file to write.

        """
        data = {
            "base_tokens": self._base_tokens,
            "merges": [list(m) for m in self.merges],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_merges(self, path: str | PathLike) -> Tokenizer:
        """Load BPE merges from a JSON file.

        Parameters
        ----------
        path : str or PathLike
            A JSON file written by `save_merges()`.

        Returns
        -------
        Tokenizer
            A new tokenizer with the loaded merges. This tokenizer is
            not changed.

        """
        data = json.loads(Path(path).read_text())
        if data["base_tokens"] != self._base_tokens:
            raise ValueError(
                "The merges were learned with a different vocabulary than "
                "this tokenizer."
            )

        return self._with_merges(tuple(m) for m in data["merges"])

    @abstractmethod
    def split(self, sequence: str) -> list[str]:
        """Split a sequence into the constituent string tokens."""

    def tokenize(
        self,
        sequences: Iterable[str] | str,
        add_start: bool = False,
        add_stop: bool = False,
        to_strings: bool = False,
        bpe_dropout: float | None = None,
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
        bpe_dropout : float, optional
            The probability of skipping each BPE merge. If `None`, the
            `bpe_dropout` attribute is used.

        Returns
        -------
        torch.tensor of shape (n_sequences, max_length) or list[list[str]]
            Either a tensor containing the integer values for each
            token, padded with 0's, or the list of tokens comprising
            each sequence.

        """
        add_start = add_start and self.start_token is not None
        add_stop = add_stop and self.stop_token is not None
        if bpe_dropout is None:
            bpe_dropout = self.bpe_dropout

        try:
            out = []
            for seq in utils.listify(sequences):
                tokens = self._apply_merges(self.split(seq), bpe_dropout)
                if add_start and tokens[0] != self.start_token:
                    tokens.insert(0, self.start_token)

                if add_stop and tokens[-1] != self.stop_token:
                    tokens.append(self.stop_token)

                if to_strings:
                    out.append(tokens)
                    continue

                out.append([self.index[t] for t in tokens])

            if to_strings:
                return out
        except KeyError as err:
            raise ValueError("Unrecognized token") from err

        return _pad_tokens(out)

    def detokenize(
        self,
        tokens: torch.Tensor,
        join: bool = True,
        trim_start_token: bool = True,
        trim_stop_token: bool = True,
        expand: bool = True,
    ) -> list[str] | list[list[str]]:
        """Retrieve sequences from tokens.

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
        expand : bool, optional
            Expand tokens created by BPE merges into the tokens returned
            by `split()`.

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

                if expand:
                    seq.extend(self.expansions[idx])
                else:
                    seq.append(self.reverse_index[idx])

            if trim_start_token and seq[0] == self.start_token:
                seq.pop(0)

            if join:
                seq = "".join(seq)

            decoded.append(seq)

        return decoded


def _merge_tokens(
    tokens: list[str],
    ranks: dict[tuple[str, str], int],
    dropout: float,
) -> list[str]:
    """Merge the adjacent tokens of a sequence with BPE.

    The adjacent pair with the lowest rank is merged until no pairs can
    be merged. When pairs have the same rank, the leftmost is merged.

    Parameters
    ----------
    tokens : list[str]
        The tokens of the sequence.
    ranks : dict[tuple[str, str], int]
        The order in which each pair of tokens is merged.
    dropout : float
        The probability of skipping each merge.

    Returns
    -------
    list[str]
        The merged tokens.

    """
    out = list(tokens)
    while len(out) > 1:
        best = None
        best_rank = len(ranks)
        for i, pair in enumerate(zip(out, out[1:])):
            rank = ranks.get(pair)
            if rank is None or rank >= best_rank:
                continue

            if dropout and random.random() < dropout:
                continue

            best, best_rank = i, rank

        if best is None:
            break

        out[best : best + 2] = [out[best] + out[best + 1]]

    return out


class _PairCounter:
    """Count the pairs of adjacent tokens while learning BPE merges.

    Each merge only updates the sequences that contain the merged pair.

    Parameters
    ----------
    counts : Counter[tuple[str, ...]]
        The number of times that each tokenized sequence occurs.
    lengths : dict[str, int]
        The number of tokens returned by `split()` in each token.
    special : set[str | None]
        Special tokens, which are never merged.
    max_token_length : int, optional
        The maximum number of tokens returned by `split()` in a merged
        token.

    """

    def __init__(
        self,
        counts: Counter,
        lengths: dict[str, int],
        special: set[str | None],
        max_token_length: int | None,
    ) -> None:
        """Initialize a _PairCounter."""
        self.lengths = lengths
        self.special = special
        self.max_token_length = max_token_length
        self.banned = set()
        self.words = [list(w) for w in counts]
        self.freqs = list(counts.values())
        self.pair_counts = Counter()
        self.where = defaultdict(set)
        for i, (word, freq) in enumerate(zip(self.words, self.freqs)):
            for pair in zip(word, word[1:]):
                if self._valid(pair):
                    self.pair_counts[pair] += freq
                    self.where[pair].add(i)

        self.heap = [(-c, p) for p, c in self.pair_counts.items()]
        heapq.heapify(self.heap)

    def _valid(self, pair: tuple[str, str]) -> bool:
        """Check whether a pair of tokens can be merged.

        Parameters
        ----------
        pair : tuple[str, str]
            The pair of tokens.

        Returns
        -------
        bool
            Whether the pair can be merged.

        """
        if pair in self.banned or self.special.intersection(pair):
            return False

        if self.max_token_length is None:
            return True

        length = self.lengths[pair[0]] + self.lengths[pair[1]]
        return length <= self.max_token_length

    def pop_best(self, min_frequency: int) -> tuple[str, str] | None:
        """Get the most frequent pair of tokens.

        Ties are broken by the lexical order of the pairs.

        Parameters
        ----------
        min_frequency : int
            The minimum number of times that the pair must occur.

        Returns
        -------
        tuple[str, str] or None
            The most frequent pair, or `None` if no pair occurs at least
            `min_frequency` times.

        """
        while self.heap:
            neg_count, pair = heapq.heappop(self.heap)
            if self.pair_counts.get(pair, 0) != -neg_count:
                continue  # A stale entry.

            if -neg_count < min_frequency:
                return None

            return pair

        return None

    def ban(self, pair: tuple[str, str]) -> None:
        """Prevent a pair of tokens from being merged.

        Parameters
        ----------
        pair : tuple[str, str]
            The pair of tokens.

        """
        self.banned.add(pair)
        self.pair_counts.pop(pair, None)

    def merge(self, pair: tuple[str, str]) -> None:
        """Merge a pair of tokens in every sequence and update the counts.

        Parameters
        ----------
        pair : tuple[str, str]
            The pair of tokens to merge.

        """
        left, right = pair
        merged = left + right
        self.lengths[merged] = self.lengths[left] + self.lengths[right]
        changed = set()
        for i in self.where.pop(pair, ()):
            self._merge_word(i, left, right, changed)

        self.pair_counts.pop(pair, None)
        changed.discard(pair)
        for changed_pair in changed:
            count = self.pair_counts[changed_pair]
            if count > 0:
                heapq.heappush(self.heap, (-count, changed_pair))
            else:
                del self.pair_counts[changed_pair]

    def _merge_word(
        self,
        i: int,
        left: str,
        right: str,
        changed: set[tuple[str, str]],
    ) -> None:
        """Merge a pair of tokens in one sequence and update the counts.

        Only the pairs next to each merged pair are updated.

        Parameters
        ----------
        i : int
            The index of the sequence.
        left : str
            The left token of the pair.
        right : str
            The right token of the pair.
        changed : set[tuple[str, str]]
            The pairs whose counts changed, which is updated in place.

        """
        word, freq = self.words[i], self.freqs[i]
        merged = left + right
        last = len(word) - 1
        new_word = []
        prev_end = -1
        j = 0
        while j <= last:
            if not (j < last and word[j] == left and word[j + 1] == right):
                new_word.append(word[j])
                j += 1
                continue

            if j:
                if prev_end != j:  # Not already removed by the last merge.
                    self._update((word[j - 1], left), -freq, i, changed)

                self._update((new_word[-1], merged), freq, i, changed)

            if j + 2 <= last:
                self._update((right, word[j + 2]), -freq, i, changed)
                if not (
                    j + 3 <= last
                    and word[j + 2] == left
                    and word[j + 3] == right
                ):
                    self._update((merged, word[j + 2]), freq, i, changed)

            new_word.append(merged)
            j += 2
            prev_end = j

        self.words[i] = new_word

    def _update(
        self,
        pair: tuple[str, str],
        delta: int,
        i: int,
        changed: set[tuple[str, str]],
    ) -> None:
        """Change the count of a pair of tokens.

        Parameters
        ----------
        pair : tuple[str, str]
            The pair of tokens.
        delta : int
            The change in the count.
        i : int
            The index of the sequence that contains the pair.
        changed : set[tuple[str, str]]
            The pairs whose counts changed, which is updated in place.

        """
        if not self._valid(pair):
            return

        self.pair_counts[pair] += delta
        if delta > 0:
            self.where[pair].add(i)

        changed.add(pair)


def _pad_tokens(tokens: list[list[int]]) -> torch.Tensor:
    """Pad integerized tokens into a single tensor.

    Parameters
    ----------
    tokens : list of list of int
        The integerized tokens for each sequence.

    Returns
    -------
    torch.Tensor of shape (n_sequences, max_length)
        The tokens, padded with 0's.

    """
    lengths = torch.tensor([len(t) for t in tokens], dtype=torch.int64)
    max_length = int(lengths.max()) if len(tokens) else 0
    padded = torch.zeros((len(tokens), max_length), dtype=torch.int64)
    mask = torch.arange(max_length) < lengths[:, None]
    padded[mask] = torch.tensor(
        [t for seq in tokens for t in seq], dtype=torch.int64
    )
    return padded
