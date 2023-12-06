"""Tokenizers for peptides."""
from __future__ import annotations

import re
from collections.abc import Iterable

import numba as nb
import numpy as np
import torch
from pyteomics.proforma import GenericModification, MassModification

from .. import utils
from ..constants import H2O, PROTON
from ..primitives import MSKB_TO_UNIMOD, Peptide, PeptideIons
from .tokenizer import Tokenizer


class PeptideTokenizer(Tokenizer):
    """A tokenizer for ProForma peptide sequences.

    Parse and tokenize ProForma-compliant peptide sequences.

    residues : iterable of str
        Residues and modifications to add to the vocabulary byone
        the standard 20 amino acids.
    replace_isoleucine_with_leucine : bool
        Replace I with L residues, because they are isobaric and often
        indistinguishable by mass spectrometry.
    reverse : bool
        Reverse the sequence for tokenization, C-terminus to N-terminus.
    """

    residues = set("ACDEFGHIKLMNPQRSTVWY")

    # The peptide parsing function:
    _parse_peptide = Peptide.from_proforma

    def __init__(
        self,
        residues: Iterable[str] | None = None,
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
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
                try:
                    del self.residues["I"]
                except TypeError:
                    self.residues.remove("I")

        super().__init__(self.residues)

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

    @classmethod
    def from_proforma(
        cls,
        sequences: Iterable[str],
        replace_isoleucine_with_leucine: bool = True,
        reverse: bool = True,
    ) -> PeptideTokenizer:
        """Create a tokenizer with the observed peptide modications.

        Modifications are parsed from ProForma 2.0-compliant peptide strings
        and added to the vocabulary.

        Parameters
        ----------
        sequences : Iterable[str]
            The peptides from which to parse modifications.
        replace_isoleucine_with_leucine : bool
            Replace I with L residues, because they are isobaric and often
            indistinguishable by mass spectrometry.
        reverse : bool
            Reverse the sequence for tokenization, C-terminus to N-terminus.

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

                if token == "-":
                    continue

                try:
                    res, mod = re.search(r"(.*)\[(.*)\]", token).groups()
                    try:
                        mod_mass = MassModification(mod).mass
                    except ValueError:
                        mod_mass = GenericModification(mod).mass
                except AttributeError:
                    res, mod_mass = token, 0.0

                try:
                    res_mass = cls.residues.get(res, 0)
                except KeyError as err:
                    raise ValueError(f"Unrecognized token {token}.") from err
                except AttributeError:
                    res_mass = 0.0  # In case we don't care about ions.

                new_res[token] = res_mass + mod_mass

        return cls(new_res, replace_isoleucine_with_leucine, reverse)

    @staticmethod
    def from_massivekb(
        replace_isoleucine_with_leucine: bool = True,
        reverse: bool = True,
    ) -> MskbPeptideTokenizer:
        """Create a tokenizer with the observed peptide modications.

        Modifications are parsed from MassIVE-KB peptide strings
        and added to the vocabulary.

        Parameters
        ----------
        replace_isoleucine_with_leucine : bool
            Replace I with L residues, because they are isobaric and often
            indistinguishable by mass spectrometry.
        reverse : bool
            Reverse the sequence for tokenization, C-terminus to N-terminus.

        Returns
        -------
        MskbPeptideTokenizer
            A tokenizer for peptides with the observed modifications.
        """
        return MskbPeptideTokenizer.from_proforma(
            [f"{mod}A" for mod in MSKB_TO_UNIMOD.values()],
            replace_isoleucine_with_leucine,
            reverse,
        )


class PeptideIonTokenizer(PeptideTokenizer):
    """A tokenizer for ProForma peptide precursor ions.

    Parse and tokenize ProForma-compliant peptide sequences. Additionally,
    use this class to calculate fragment and precursor ion masses.

    Parameters
    ----------
    residues : dict of [str, float], optional
        Residues and modifications to add to the vocabulary beyond the
        standard 20 amino acids.
    replace_isoleucine_with_leucine : bool
        Replace I with L residues, because they are isobaric and often
        indistinguishable by mass spectrometry.
    reverse : bool
        Reverse the sequence for tokenization, C-terminus to N-terminus.

    Attributes
    ----------
    residues : numba.typed.Dict[str, float]
        The residues and modifications and their associated masses.
        terminal modifcations are indicated by `-`.
    index : SortedDict{str, int}
        The mapping of residues and modifications to integer representations.
    reverse_index : list[None | str]
        The ordered residues and modifications where the list index is the
        integer representation for a token.
    stop_token : str
        The stop token.
    """

    residues = nb.typed.Dict.empty(
        nb.types.unicode_type,
        nb.types.float64,
    )
    residues.update(
        G=57.021463735,
        A=71.037113805,
        S=87.032028435,
        P=97.052763875,
        V=99.068413945,
        T=101.047678505,
        C=103.009184505,
        L=113.084064015,
        I=113.084064015,
        N=114.042927470,
        D=115.026943065,
        Q=128.058577540,
        K=128.094963050,
        E=129.042593135,
        M=131.040484645,
        H=137.058911875,
        F=147.068413945,
        R=156.101111050,
        Y=163.063328575,
        W=186.079312980,
    )

    def __init__(
        self,
        residues: dict[str, float] | None = None,
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
    ) -> None:
        """Initialize a PeptideTokenizer."""
        super().__init__(
            residues=residues,
            replace_isoleucine_with_leucine=replace_isoleucine_with_leucine,
            reverse=reverse,
        )

    def __getstate__(self) -> dict:
        """How to pickle the object."""
        self.residues = dict(self.residues)
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        """How to unpickle the object."""
        self.__dict__ = state
        residues = self.residues
        self.residues = nb.typed.Dict.empty(
            nb.types.unicode_type,
            nb.types.float64,
        )
        self.residues.update(residues)

    def ions(  # noqa: C901
        self,
        sequences: Iterable[str],
        precursor_charges: Iterable[int] | str,
        max_fragment_charge: int | None = None,
    ) -> tuple[torch.Tensor[float], list[torch.Tensor[float]]]:
        """Calculate the m/z for the precursor and fragment ions.

        Currently depthcharge only support b and y ions.

        Parameters
        ----------
        sequences : Iterable[str],
            The peptide sequences.
        precursor_charges : Iterable[int] or None, optional
            The charge of each precursor ion. If ``None``, the charge state
            is expected to be found in the peptide strings.
        max_fragment_charge : int or None, optional
            The maximum charge for fragment ions. The default is to consider
            up to the ``max(precursor_charge - 1, 1)``.

        Returns
        -------
        list of PeptideIons
            The precursor and fragment ions generated by the peptide.
        """
        sequences = utils.listify(sequences)
        if max_fragment_charge is None:
            max_fragment_charge = np.inf

        if precursor_charges is None:
            precursor_charges = [None] * len(sequences)
        else:
            precursor_charges = utils.listify(precursor_charges)

        if len(sequences) != len(precursor_charges):
            raise ValueError(
                "The number of sequences and precursor charges did not match."
            )

        out = []
        for seq, charge in zip(sequences, precursor_charges):
            if isinstance(seq, str):
                if self.replace_isoleucine_with_leucine:
                    seq = seq.replace("I", "L")

                try:
                    pep = Peptide.from_proforma(seq)
                except ValueError:
                    pep = Peptide.from_massivekb(seq)

                tokens = pep.split()
                if charge is None:
                    charge = max(pep.charge - 1, 1)
            else:
                tokens = seq

            if charge is None:
                raise ValueError(
                    f"No charge was provided for {seq}",
                )

            try:
                prec = _calc_precursor_mass(
                    nb.typed.List(tokens),
                    charge,
                    self.residues,
                )
            except KeyError as err:
                raise ValueError(
                    f"Unrecognized token(s) in {''.join(tokens)}"
                ) from err

            frags = _calc_fragment_masses(
                nb.typed.List(tokens),
                min(charge, max_fragment_charge),
                self.residues,
            )

            ions = PeptideIons(
                tokens=tokens,
                precursor=prec,
                fragments=torch.tensor(frags),
            )

            out.append(ions)

        return out

    @staticmethod
    def from_massivekb(
        replace_isoleucine_with_leucine: bool = True,
        reverse: bool = True,
    ) -> MskbPeptideTokenizer:
        """Create a tokenizer with the observed peptide modications.

        Modifications are parsed from MassIVE-KB peptide strings
        and added to the vocabulary.

        Parameters
        ----------
        replace_isoleucine_with_leucine : bool
            Replace I with L residues, because they are isobaric and often
            indistinguishable by mass spectrometry.
        reverse : bool
            Reverse the sequence for tokenization, C-terminus to N-terminus.

        Returns
        -------
        MskbPeptideTokenizer
            A tokenizer for peptides with the observed modifications.
        """
        return MskbPeptideIonTokenizer.from_proforma(
            [f"{mod}A" for mod in MSKB_TO_UNIMOD.values()],
            replace_isoleucine_with_leucine,
            reverse,
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
    stop_token : str
        The stop token.

    """

    _parse_peptide = Peptide.from_massivekb


class MskbPeptideIonTokenizer(PeptideIonTokenizer):
    """A tokenizer for MassIVE-KB peptide precursor ions.

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
    stop_token : str
        The stop token.

    """

    _parse_peptide = Peptide.from_massivekb


@nb.njit
def _calc_precursor_mass(
    tokens: list[str],
    charge: int,
    masses: nb.typed.Dict,
) -> float:
    """Calculate the precursor mass of a peptide sequence.

    Parameters
    ----------
    tokens : list of str
        The tokenized peptide sequence.
    charge : int
        The charge state to consider. Use 'None' to get the neutral mass.
    masses : nb.typed.Dict
        The mass dictionary to use.

    Returns
    -------
    float
        The precurosr monoisotopic m/z.
    """
    mass = sum([masses[t] for t in tokens]) + H2O
    if charge is not None:
        mass = _mass2mz(mass, charge)

    return mass


@nb.njit
def _calc_fragment_masses(
    tokens: list[str],
    charge: int,
    masses: nb.typed.Dict,
) -> np.ndarray[float]:
    """Calculate the b and y ions for a peptide sequence.

    Parameters
    ----------
    tokens : list of str
        The tokenized peptide sequence.
    charge : int
        The charge state to consider. Use 'None' to get the neutral mass.
    masses : nb.typed.Dict
        The mass dictionary to use.

    Returns
    -------
    np.ndarray of shape (2, len(seq) - 1, charge)
        The m/z of the predicted b and y ions.
    """
    # Handle terminal mods:
    seq = np.empty(len(tokens))
    n_mod = False
    c_mod = False
    for idx, token in enumerate(tokens):
        if not idx and token.endswith("-"):
            n_mod = True

        if idx == (len(tokens) - 1) and token.startswith("-"):
            c_mod = True

        seq[idx] = masses[token]

    if n_mod:
        seq[1] += seq[0]
        seq = seq[1:]

    if c_mod:
        seq[-2] += seq[-1]
        seq = seq[:-1]

    # Calculate fragments:
    max_charge = min(charge, 2)
    n_ions = len(seq) - 1
    ions = np.empty((2, n_ions, max_charge))
    b_mass = 0
    y_mass = H2O
    for idx in range(n_ions):
        b_mass += seq[idx]
        y_mass += seq[-(idx + 1)]
        for cur_charge in range(1, max_charge + 1):
            z_idx = cur_charge - 1
            ions[0, idx, z_idx] = _mass2mz(b_mass, cur_charge)
            ions[1, idx, z_idx] = _mass2mz(y_mass, cur_charge)

    return ions


@nb.njit
def _mass2mz(mass: float, charge: int) -> float:
    """Calculate the m/z.

    Parameters
    ----------
    mass : float
        The neutral mass.
    charge : int
        The charge.

    Returns
    -------
    float
       The m/z
    """
    return (mass / charge) + PROTON
