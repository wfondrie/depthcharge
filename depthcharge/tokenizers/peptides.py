"""Tokenizers for peptides."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from pyteomics.proforma import GenericModification, MassModification

from .. import utils
from ..constants import H2O, PROTON
from ..primitives import MSKB_TO_UNIMOD, Peptide
from .tokenizer import Tokenizer


class PeptideTokenizer(Tokenizer):
    """A tokenizer for ProForma peptide sequences.

    Parse and tokenize ProForma-compliant peptide sequences.

    Parameters
    ----------
    residues : dict[str, float], optional
        Residues and modifications to add to the vocabulary beyond
        the standard 20 amino acids.
    replace_isoleucine_with_leucine : bool, optional
        Replace I with L residues, because they are isomeric and often
        indistinguishable by mass spectrometry.
    reverse : bool, optional
        Reverse the sequence for tokenization, C-terminus to N-terminus.
    start_token : str, optional
        The start token to use.
    stop_token : str, optional
        The stop token to use.

    Attributes
    ----------
    residues : SortedDict[str, float]
        The residues and modifications and their associated masses.
        terminal modifcations are indicated by `-`.
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

    residues = {
        "G": 57.021463735,
        "A": 71.037113805,
        "S": 87.032028435,
        "P": 97.052763875,
        "V": 99.068413945,
        "T": 101.047678505,
        "C": 103.009184505,
        "L": 113.084064015,
        "I": 113.084064015,
        "N": 114.042927470,
        "D": 115.026943065,
        "Q": 128.058577540,
        "K": 128.094963050,
        "E": 129.042593135,
        "M": 131.040484645,
        "H": 137.058911875,
        "F": 147.068413945,
        "R": 156.101111050,
        "Y": 163.063328575,
        "W": 186.079312980,
    }

    # The peptide parsing function:
    _parse_peptide = Peptide.from_proforma

    def __init__(
        self,
        residues: Iterable[str] | None = None,
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
        start_token: str | None = None,
        stop_token: str | None = "$",
    ) -> None:
        """Initialize a PeptideTokenizer."""
        self.replace_isoleucine_with_leucine = replace_isoleucine_with_leucine
        self.reverse = reverse

        # Note that these also secretly work on dicts too ;)
        self.residues = self.residues.copy()
        if residues is not None:
            self.residues.update(residues)

        if self.replace_isoleucine_with_leucine:
            if "I" in self.residues:
                del self.residues["I"]

        super().__init__(self.residues, start_token, stop_token)
        self.masses = torch.tensor(
            [self.residues.get(a, 0.0) for a in self.reverse_index]
        )

    def calculate_precursor_ions(
        self,
        tokens: torch.Tensor | Iterable[str],
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the m/z for precursor ions.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_sequences, len_seq)
            The tokens corresponding to the peptide sequence.
        charges : torch.Tensor of shape (n_sequences,)
            The charge state for each peptide.

        Returns
        -------
        torch.Tensor
            The monoisotopic m/z for each charged peptide.

        """
        if isinstance(tokens[0], str):
            tokens = self.tokenize(utils.listify(tokens))

        if not isinstance(charges, torch.Tensor):
            charges = torch.tensor(charges)
            if not charges.shape:
                charges = charges[None]

        masses = self.masses[tokens].sum(dim=1) + H2O
        return (masses / charges) + PROTON

    def split(self, sequence: str) -> list[str]:
        """Split a ProForma peptide sequence.

        Parameters
        ----------
        sequence : str
            The peptide sequence.

        Returns
        -------
        list[str]
            The tokens that comprise the peptide sequence.

        """
        pep = self._parse_peptide(sequence)
        if self.replace_isoleucine_with_leucine:
            pep.sequence = pep.sequence.replace("I", "L")

        pep = pep.split()
        if self.reverse:
            pep.reverse()

        return pep

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
            Remove the stop token from the end of a sequence.

        Returns
        -------
        list[str] or list[list[str]]
            The decoded sequences each as a string or list or strings.

        """
        decoded = super().detokenize(
            tokens=tokens,
            join=False,
            trim_start_token=trim_start_token,
            trim_stop_token=trim_stop_token,
        )

        if self.reverse:
            decoded = [d[::-1] for d in decoded]

        if join:
            decoded = ["".join(pep) for pep in decoded]

        return decoded

    @classmethod
    def from_proforma(
        cls,
        sequences: Iterable[str],
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
        start_token: str | None = None,
        stop_token: str | None = "$",
    ) -> PeptideTokenizer:
        """Create a tokenizer with the observed peptide modications.

        Modifications are parsed from ProForma 2.0-compliant peptide strings
        and added to the vocabulary.

        Parameters
        ----------
        sequences : Iterable[str]
            The peptides from which to parse modifications.
        replace_isoleucine_with_leucine : bool, optional
            Replace I with L residues, because they are isobaric and often
            indistinguishable by mass spectrometry.
        reverse : bool, optional
            Reverse the sequence for tokenization, C-terminus to N-terminus.
        start_token : str, optional
            The start token to use.
        stop_token : str, optional
            The stop token to use.

        Returns
        -------
        PeptideTokenizer
            A tokenizer for peptides with the observed modifications.

        """
        if isinstance(sequences, str):
            sequences = [sequences]

        # Parse modifications:
        new_res = {}
        for peptide in sequences:
            parsed = Peptide.from_proforma(peptide).split()
            for token in parsed:
                if token in cls.residues:
                    continue

                try:
                    res, mod = re.search(r"(.*)\[(.*)\]", token).groups()
                    try:
                        mod_mass = MassModification(mod).mass
                    except ValueError:
                        mod_mass = GenericModification(mod).mass
                except AttributeError as err:
                    raise KeyError(f"Unknown residue {token}") from err

                try:
                    res_mass = cls.residues.get(res, 0)
                except KeyError as err:
                    raise ValueError(f"Unrecognized token {token}.") from err
                except AttributeError:
                    res_mass = 0.0  # In case we don't care about ions.

                new_res[token] = res_mass + mod_mass

        return cls(
            new_res,
            replace_isoleucine_with_leucine,
            reverse,
            start_token,
            stop_token,
        )

    @staticmethod
    def from_massivekb(
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
        start_token: str | None = None,
        stop_token: str | None = "$",
    ) -> MskbPeptideTokenizer:
        """Create a tokenizer with the observed peptide modications.

        Modifications are parsed from MassIVE-KB peptide strings
        and added to the vocabulary.

        Parameters
        ----------
        replace_isoleucine_with_leucine : bool, optional
            Replace I with L residues, because they are isobaric and often
            indistinguishable by mass spectrometry.
        reverse : bool, optional
            Reverse the sequence for tokenization, C-terminus to N-terminus.
        start_token : str, optional
            The start token to use.
        stop_token : str, optional
            The stop token to use.

        Returns
        -------
        MskbPeptideTokenizer
            A tokenizer for peptides with the observed modifications.

        """
        return MskbPeptideTokenizer.from_proforma(
            [f"{mod}A" for mod in MSKB_TO_UNIMOD.values()],
            replace_isoleucine_with_leucine,
            reverse,
            start_token,
            stop_token,
        )


class MskbPeptideTokenizer(PeptideTokenizer):
    """A tokenizer for MassIVE-KB peptide sequences.

    MassIVE-KB includes N-term carbamylation, NH3-loss, acetylation, as well as
    M oxidation, and deamidation of N and Q, in a manner that does not comply
    with the ProForma standard.

    Parameters
    ----------
    replace_isoleucine_with_leucine : bool
        Replace I with L residues, because they are isobaric and often
        indistinguishable by mass spectrometry.
    reverse : bool
        Reverse the sequence for tokenization, C-terminus to N-terminus.
    start_token : str, optional
        The start token to use.
    stop_token : str, optional
        The stop token to use.

    Attributes
    ----------
    residues : SortedDict[str, float]
        The residues and modifications and their associated masses.
        terminal modifcations are indicated by `-`.
    index : SortedDicte{str, int}
        The mapping of residues and modifications to integer representations.
    reverse_index : list[None | str]
        The ordered residues and modifications where the list index is the
        integer representation for a token.
    start_int : int
        The integer representation of the start token
    stop_int : int
        The integer representation of the stop token.
    padding_int : int
        The integer used to represent padding.

    """

    _parse_peptide = Peptide.from_massivekb
