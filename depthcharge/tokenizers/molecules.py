"""Tokenizers for small molecules."""

from __future__ import annotations

from collections.abc import Iterable

import selfies as sf

from .. import utils
from ..primitives import Molecule
from .tokenizer import Tokenizer


class MoleculeTokenizer(Tokenizer):
    """A tokenizer for small molecules.

    Tokenize SMILES and SELFIES representations of small molecules.
    SMILES are internally converted to SELFIES representations.

    Parameters
    ----------
    selfies_vocab : Iterable[str]
        The SELFIES tokens to be considered.
    start_token : str, optional
        The start token to use.
    stop_token : str, optional
        The stop token to use.

    Attributes
    ----------
    index : SortedDict{str, int}
        The mapping of residues and modifications to integer representations.
    reverse_index : list[None | str]
        The ordered residues and modifications where the list index is the
        integer representation for a token.
    start_token : str
        The start token
    stop_token : str
        The stop token.
    start_int : int
        The integer representation of the start token
    stop_int : int
        The integer representation of the stop token.
    padding_int : int
        The integer used to represent padding.

    """

    def __init__(
        self,
        selfies_vocab: Iterable[str] | None = None,
        start_token: str | None = None,
        stop_token: str | None = "$",
    ) -> None:
        """Initialize a MoleculeTokenizer."""
        if selfies_vocab is None:
            selfies_vocab = sf.get_semantic_robust_alphabet()

        self.selfies_vocab = selfies_vocab
        super().__init__(selfies_vocab, start_token, stop_token)

    def split(self, sequence: str) -> list[str]:
        """Split a SMILES or SELFIES string into SELFIES tokens.

        Parameters
        ----------
        sequence : str
            The SMILES or SELFIES string representing a molecule.

        Returns
        -------
        List[str]
            The SELFIES tokens representing the molecule.

        """
        try:
            return list(sf.split_selfies(sf.encoder(sequence)))
        except sf.EncoderError:
            return list(sf.split_selfies(sequence))

    @classmethod
    def from_smiles(
        cls,
        smiles: Iterable[str] | str,
        start_token: str | None = None,
        stop_token: str | None = "$",
    ) -> MoleculeTokenizer:
        """Learn the vocabulary from SMILES strings.

        Parameters
        ----------
        smiles : Iterable[str] | str
            Create a vocabulary from all unique tokens in these SMILES strings.
        start_token : str, optional
            The start token to use.
        stop_token : str, optional
            The stop token to use.

        Returns
        -------
        MoleculeTokenizer
            The tokenizer restricted to the vocabulary present in the
            input SMILES strings.

        """
        vocab = sf.get_alphabet_from_selfies(
            Molecule(s).to_selfies() for s in utils.listify(smiles)
        )

        return cls(vocab, start_token, stop_token)

    @classmethod
    def from_selfies(
        cls,
        selfies: Iterable[str] | str,
        start_token: str | None = None,
        stop_token: str | None = "$",
    ) -> MoleculeTokenizer:
        """Learn the vocabulary from SELFIES strings.

        Parameters
        ----------
        selfies : Iterable[str] | str
            Create a vocabulary from all unique tokens in these SELFIES
            strings.
        start_token : str, optional
            The start token to use.
        stop_token : str, optional
            The stop token to use.

        Returns
        -------
        MoleculeTokenizer
            The tokenizer restricted to the vocabulary present in the
            input SMILES strings.

        """
        vocab = sf.get_alphabet_from_selfies(utils.listify(selfies))
        return cls(vocab, start_token, stop_token)
