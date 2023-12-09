"""Tokenizers for small molecules."""
from __future__ import annotations

from collections.abc import Iterable

import selfies as sf

from .. import utils
from ..primitives import Molecule
from .tokenizer import Tokenizer


class MoleculeTokenizer(Tokenizer):
    """A tokenizer for small molecules.

    SMILES strings representing small molecules are parsed as
    SELFIES representations and split into tokens.

    Parameters
    ----------
    selfies_vocab : Iterable[str]
        The SELFIES tokens to be considered.
    """

    def __init__(self, selfies_vocab: Iterable[str] | None = None) -> None:
        """Initialize a MoleculeTokenizer."""
        if selfies_vocab is None:
            selfies_vocab = sf.get_semantic_robust_alphabet()

        self.selfies_vocab = selfies_vocab
        super().__init__(selfies_vocab)

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
            return sf.split_selfies(sequence)
        except AttributeError:
            return sf.split_selfies(Molecule(sequence).to_selfies())

    @classmethod
    def from_smiles(cls, smiles: Iterable[str] | str) -> MoleculeTokenizer:
        """Learn the vocabulary from SMILES strings.

        Parameters
        ----------
        smiles : Iterable[str] | str
            Create a vocabulary from all unique tokens in these SMILES strings.

        Returns
        -------
        MoleculeTokenizer
            The tokenizer restricted to the vocabulary present in the
            input SMILES strings.
        """
        vocab = sf.get_alphabet_from_selfies(
            Molecule(s).to_selfies() for s in utils.listify(smiles)
        )

        return cls(vocab)

    @classmethod
    def from_selfies(cls, selfies: Iterable[str] | str) -> MoleculeTokenizer:
        """Learn the vocabulary from SELFIES strings.

        Parameters
        ----------
        selfies : Iterable[str] | str
            Create a vocabulary from all unique tokens in these SELFIES
            strings.

        Returns
        -------
        MoleculeTokenizer
            The tokenizer restricted to the vocabulary present in the
            input SMILES strings.
        """
        vocab = sf.get_alphabet_from_selfies(utils.listify(selfies))
        return cls(vocab)
