"""Tokenizers for peptides."""
from __future__ import annotations

from collections.abc import Iterable

import numba as nb
import numpy as np
import torch
from spectrum_utils import proforma

from .. import utils
from ..constants import C13, H2O, PROTON
from ..primitives import MSKB_TO_UNIMOD, Peptide
from .tokenizer import Tokenizer


class PeptideTokenizer(Tokenizer):
    """A tokenizer for ProForma peptide sequences.

    Parameters
    ----------
    residues : dict[str, float], optional
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

    # The peptide parsing function:
    _parse_peptide = Peptide.from_proforma

    def __init__(
        self,
        residues: dict[str, float] | None = None,
        replace_isoleucine_with_leucine: bool = True,
        reverse: bool = True,
    ) -> None:
        """Initialize a PeptideTokenizer."""
        self.replace_isoleucine_with_leucine = replace_isoleucine_with_leucine
        self.reverse = reverse
        self.residues = self.residues.copy()
        if residues is not None:
            self.residues.update(residues)

        if self.replace_isoleucine_with_leucine:
            del self.residues["I"]

        super().__init__(list(self.residues.keys()))

    def split(self, sequence: str) -> list[str]:
        """Split a ProForma peptide sequence."""
        pep = self._parse_peptide(sequence)
        if self.replace_isoleucine_with_leucine:
            pep.sequence = pep.sequence.replace("I", "L")

        pep = pep.split()
        if self.reverse:
            pep.reverse()

        return pep

    def precursor_mz(
        self,
        sequences: Iterable[str],
        charges: Iterable[int | None] | None = None,
        n_isotopes: int = 1,
    ) -> torch.Tensor[float]:
        """Calculate the precursor m/z for peptide sequences.

        Parameters
        ----------
        sequences : Iterable[str],
            The peptide sequences.
        charges : Iterable[int | None] | None = None, optional
            The charge states. If ``None`` then charge states are
            expected to be found in the peptide string.
        n_isotopes : int, optional
            The number of C13 isotopes to include. ``1`` calculates
            only the monoisotpic m/z.

        Returns
        -------
        torch.Tensor of shape (n_sequences, n_isotopes)
            A tensor containing the precurosr m/z values.
        """
        sequences = utils.listify(sequences)
        if charges is None:
            charges = [None] * len(sequences)

        out = torch.empty(len(sequences), n_isotopes)
        for idx, seq, charge in enumerate(zip(sequences, charges)):
            if isinstance(seq, str):
                try:
                    pep = Peptide.from_proforma(seq)
                except ValueError:
                    pep = Peptide.from_massivekb(seq)

                tokens = pep.split()
                if charge is None:
                    charge = pep.charge
            else:
                tokens = seq

            if charge is None:
                raise ValueError(
                    f"No charge was provided for {seq}",
                )

            out[idx, :] = _calc_precursor_mass(
                tokens,
                charge,
                n_isotopes,
                self.residues,
            )

        return out

    def fragment_mz(
        self,
        sequences: Iterable[str],
        max_charge: Iterable[int | None] | int | None = 1,
    ) -> list[torch.Tensor[float]]:
        """Calculate the precursor m/z for peptide sequences.

        Parameters
        ----------
        sequences : Iterable[str],
            The peptide sequences.
        max_charge : Iterable[int | None] | int | None, optional
            The charge states. If ``None`` then charge states are
            expected to be found in the peptide string.
        n_isotopes : int, optional
            The number of C13 isotopes to include. ``1`` calculates
            only the monoisotpic m/z.

        Returns
        -------
        list of torch.Tensor of shape (2, len(seq) - 1, charge)
            A tensor containing the fragment m/z values.
        """
        sequences = utils.listify(sequences)
        try:
            iter(max_charge)
        except TypeError:
            max_charge = [max_charge] * len(sequences)

        out = []
        for seq, charge in zip(sequences, max_charge):
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

            frags = _calc_fragment_masses(
                nb.typed.List(tokens),
                charge,
                self.residues,
            )

            out.append(torch.tensor(frags))

        return out

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
        new_res = cls.residues.copy()
        for peptide in sequences:
            parsed = proforma.parse(peptide)[0]
            if parsed.modifications is None:
                continue

            for mod in parsed.modifications:
                loc = mod.position
                try:
                    modstr = f"[{mod.source[0].name}]"
                except (TypeError, AttributeError):
                    modstr = f"[{mod.mass:+0.6f}]"

                if loc == "N-term":
                    modstr += "-"
                elif loc == "C-term":
                    modstr = "-" + modstr
                else:
                    res = parsed.sequence[loc]
                    if res not in cls.residues.keys():
                        raise ValueError(f"Unknown amino acid {res}.")

                    modstr = res + modstr

                new_res[modstr] = mod.mass

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


class MskbPeptideTokenizer(PeptideTokenizer):
    """A tokenizer for MassIVE-KB peptide sequences.

    MassIVE-KB includes N-term carbamylation, NH3-loss, acetylation, as well as
    M oxidation, and deamidation of N and Q, in a manner that does not comply
    with the ProForma standard.

    Parameters
    ----------
    residues : dict[str, float], optional
        Residues and modifications to add to the vocabulary beyond the
        standard 20 amino acids.
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
    n_isotopes: int,
    masses: nb.typed.Dict,
) -> np.ndarray[float]:
    """Calculate the precursor mass of a peptide sequence.

    Parameters
    ----------
    tokens : list of str
        The tokenized peptide sequence.
    charge : int
        The charge state to consider. Use 'None' to get the neutral mass.
    n_isotopes : int
        The number of C13 isotopes to return, starting from the
        monoisotopic mass.
    masses : nb.typed.Dict
        The mass dictionary to use.

    Returns
    -------
    np.ndarray
        The precurosr monoisotopic m/z and requested C13 isotopes.
    """
    out = np.empty(n_isotopes)
    mass = sum(masses[t] for t in tokens) + H2O
    for isotope in range(n_isotopes):
        if isotope:
            mass += C13

        if charge is not None:
            mass = _mass2mz(mass, charge)

        out[isotope + 1] = mass

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
    n_isotopes : int
        The number of C13 isotopes to return, starting from the
        monoisotopic mass.
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
