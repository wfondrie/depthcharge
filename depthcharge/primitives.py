"""Fundamental dataclasses for depthcharge."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import selfies as sf
import torch
from PIL.PngImagePlugin import PngImageFile
from rdkit import Chem
from rdkit.Chem import Draw
from spectrum_utils import proforma

MSKB_TO_UNIMOD = {
    "+42.011": "[Acetyl]-",
    "+43.006": "[Carbamyl]-",
    "-17.027": "[Ammonia-loss]-",
    "+43.006-17.027": "[+25.980265]-",  # Not in Unimod
    "M+15.995": "M[Oxidation]",
    "N+0.984": "N[Deamidated]",
    "Q+0.984": "Q[Deamidated]",
    "C+57.021": "C[Carbamidomethyl]",
}


@dataclass
class Peptide:
    """A peptide sequence with or without modification and/or charge.

    Parameters
    ----------
    sequence : str
        The bare amino acid sequence.
    modifications : iterable of str, float, or None, optional
        The modification at each amino acid. This should be the length of
        ``sequence`` + 2, where index 0 and -1 are used for N- and C-terminal
        modification respectively. Use ``None`` in the iterable to specify
        unmodified positions. When ``modifications`` is ``None``, it is assumed
        that no modifications are present.
    charge : int, optional
        The charge of the peptide.
    """

    sequence: str
    modifications: Sequence[str | float | None] | None = None
    charge: int | None = None

    def __post_init__(self) -> None:
        """Validate the provided parameters."""
        if self.modifications is not None:
            if len(self.modifications) != len(self.sequence) + 2:
                raise ValueError(
                    "'modifications' must be two elements longer than "
                    "'sequences' to account for terminal modifications."
                )

        self.proforma = "".join(self.split())

    def split(self) -> list[str]:
        """Split the modified peptide for tokenization."""
        if self.modifications is None:
            return list(self.sequence)

        out = []
        for idx, (aa, mod) in enumerate(
            zip(f"-{self.sequence}-", self.modifications)
        ):
            if mod is None:
                if idx and (idx < len(self.modifications) - 1):
                    out.append(aa)

                continue

            try:
                modstr = f"[{mod:+0.6f}]"
            except ValueError:
                modstr = f"[{mod}]"

            if not idx:
                out.append(f"{modstr}-")
            else:
                out.append(f"{aa}{modstr}")

        return out

    @classmethod
    def from_proforma(cls, sequence: str) -> Peptide:
        """Create a Peptide from a ProForma 2.0 string.

        Parameters
        ----------
        sequence : str
            A ProForma 2.0-compliant string.

        Returns
        -------
        Peptide
            The parsed ProForma peptide.
        """
        pep = proforma.parse(sequence)[0]
        seq = pep.sequence

        # Get the charge:
        try:
            charge = pep.charge.charge
        except AttributeError:
            charge = None

        if pep.modifications is None:
            return cls(seq, None, charge)

        # Get the modifications:
        mods = [None] * (len(seq) + 2)
        for mod in pep.modifications:
            pos = mod.position
            if pos == "N-term":
                pos = 0
            elif pos == "C-term":
                pos = -1
            else:
                pos += 1

            try:
                mods[pos] = mod.source[0].name
            except (TypeError, AttributeError):
                mods[pos] = mod.mass

        return cls(seq, mods, charge)

    @classmethod
    def from_massivekb(
        cls,
        sequence: str,
        charge: int | None = None,
    ) -> Peptide:
        """Create a Peptide from MassIVE-KB annotations.

        MassIVE-KB includes N-term carbamylation, NH3-loss, acetylation,
        as well as M oxidation, and deamidation of N and Q, in a
        manner that does not comply with the ProForma standard.

        Parameters
        ----------
        sequence : str
            The peptide sequence from MassIVE-KB
        charge : int, optional
            The charge state of the peptide.

        Returns
        -------
        Peptide
            The parsed MassIVE peptide after conversion to a ProForma
            format.
        """
        sequence = cls.massivekb_to_proforma(sequence, charge)
        return cls.from_proforma(sequence)

    @classmethod
    def massivekb_to_proforma(
        cls, sequence: str, charge: int | None = None
    ) -> str:
        """Convert a MassIVE-KB peptide sequence to ProForma.

        MassIVE-KB includes N-term carbamylation, NH3-loss, acetylation,
        as well as M oxidation, and deamidation of N and Q, in a
        manner that does not comply with ProForma.

        Parameters
        ----------
        sequence : str
            The peptide sequence from MassIVE-KB
        charge : int, optional
            The charge state of the peptide.

        Returns
        -------
        str
            The parsed MassIVE peptide after conversion to a ProForma
            format.
        """
        sequence = "".join(
            [
                MSKB_TO_UNIMOD.get(aa, aa)
                for aa in re.split(r"(?<=.)(?=[A-Z])", sequence)
            ]
        )
        if charge is not None:
            sequence += f"/{charge}"

        return sequence


@dataclass
class Molecule:
    """A representation of a molecule.

    Parameters
    ----------
    smiles : str
        A SMILES string defining the molecule.
    charge : int, optional
        The charge of the molecule.
    """

    smiles: str
    charge: int | None = None

    def __post_init__(self) -> None:
        """Validate parameters."""
        self._mol = Chem.MolFromSmiles(self.smiles)
        if self._mol is None:
            raise ValueError("The SMILES was invalid.")

        # Canonnicalize the SMILES string:
        self.smiles = Chem.MolToSmiles(self._mol)

    def show(self, **kwargs: dict) -> PngImageFile:
        """Show the molecule in 2D.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to ``rdkit.Chem.Draw.MolToImage``
        """
        return Draw.MolToImage(self._mol, **kwargs)

    def to_selfies(self) -> str:
        """Convert SMILES to a SELFIES representaion."""
        return sf.encoder(self.smiles)

    @classmethod
    def from_selfies(
        cls,
        selfies: str,
        charge: int | None = None,
    ) -> Molecule:
        """Create a molecule from a SELFIES string.

        Parameters
        ----------
        selfies : str
            The SELFIES string defining the molecule.
        charge : int, optional
            The charge of the molecule.

        Returns
        -------
        Molecule
            The parsed Molecule.
        """
        return cls(sf.decoder(selfies), charge)


@dataclass
class MassSpectrum:
    """A mass spectrum.

    Parameters
    ----------
    mzs: torch.tensor of shape (n_peaks,)
        The m/z values.
    intensities: torch.tensor of shape (n_peaks, )
        The intensity values.
    precursor: Precurosr, optional
        Precursor ion information, if applicable.
    """

    mzs: torch.tensor
    intensities: torch.tensor
    precursor: Precursor | None = None

    def __post_init__(self) -> None:
        """Validate the parameters."""
        if not isinstance(self.mzs, torch.Tensor):
            self.mzs = torch.tensor(self.mzs)

        if not isinstance(self.intensities, torch.Tensor):
            self.intensities = torch.tensor(self.intensities)

        if not self.mzs.shape == self.intensities.shape:
            raise ValueError(
                "'mzs' and 'intensities' must be the same length."
            )

    def to_tensor(self) -> torch.tensor:
        """Combine the m/z and intensities into a single tensor.

        Returns
        -------
        torch.tensor of shape (n_peaks, 2)
            The spectrum information.
        """
        return torch.vstack([self.mzs, self.intensities]).T


@dataclass
class Precursor:
    """The description of a precursor ion or window.

    Parameters
    ----------
    mz : float or 2-tuple of floats
        Either the selected m/z or the m/z isolation window.
    charge : int, optional
        The charge of the precursor ion, if known.
    """

    mz: float | tuple[float, float]
    charge: int | None = None
