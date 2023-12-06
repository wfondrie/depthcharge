"""Tokenizers for small molecules."""
from __future__ import annotations

from collections.abc import Iterable

import selfies as sf

from ..primitives import Molecule
from .tokenizer import Tokenizer


class MoleculeTokenizer(Tokenizer):
    """A tokenizer for small molecules.

    SMILES strings representing small molecules are parsed as
    SELFIES representations and split into tokens.
    """

    def __init__(self, selfies_vocab: Iterable[str] | None) -> None:
        """Initialize a MoleculeTokenizer."""
        if selfies_vocab is None:
            selfies_vocab = sf.get_semantic_robust_alphabet()

        self.selfies_vocab = selfies_vocab
        super().__init__(selfies_vocab)

    def split(self, sequence: str) -> list[str]:
        """Split a SMILES string into SELFIES tokens.

        Parameters
        ----------
        sequence : str
            The SMILES string representing a molecule.

        Returns
        -------
        List[str]
            The SELFIES tokens representing the molecule.
        """
        return sf.split_selfies(Molecule(sequence).to_selfies())
