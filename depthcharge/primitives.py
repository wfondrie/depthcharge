"""Fundamental dataclasses for depthcharge."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import selfies as sf
import torch
from numpy.typing import ArrayLike
from PIL.PngImagePlugin import PngImageFile
from pyteomics import proforma
from pyteomics.proforma import GenericModification, MassModification
from rdkit import Chem
from rdkit.Chem import Draw
from spectrum_utils.spectrum import MsmsSpectrum

from . import utils
from .constants import PROTON

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

        if self.modifications is None:
            self.modifications = [None] * (len(self.sequence) + 2)

        # Parse modifications into Pyteomics' format
        parsed = [None] * (len(self.sequence) + 2)
        for idx, mod in enumerate(self.modifications):
            if mod is None:
                continue

            parsed[idx] = [_resolve_mod(m) for m in utils.listify(mod)]

        self.modifications = parsed
        n_mod = self.modifications[0]
        c_mod = self.modifications[-1]

        self.proforma = proforma.to_proforma(
            sequence=list(zip(self.sequence, self.modifications[1:-1])),
            n_term=[n_mod] if n_mod is not None else n_mod,
            c_term=[c_mod] if c_mod is not None else c_mod,
            charge_state=self.charge,
        )

    def split(self) -> list[str]:
        """Split the modified peptide for tokenization."""
        if self.modifications is None:
            return list(self.sequence)

        out = []
        for idx, (aa, mods) in enumerate(
            zip(f"-{self.sequence}-", self.modifications)
        ):
            if mods is None:
                if idx and (idx < len(self.modifications) - 1):
                    out.append(aa)

                continue

            if len(mods) == 1:
                try:
                    modstr = f"[{mods[0].name}]"
                except (AttributeError, ValueError):
                    modstr = f"[{mods[0].mass:+0.6f}]"
            else:
                modstr = f"[{sum(m.mass for m in mods):+0.6f}]"

            if not idx:
                out.append(f"{modstr}-")
            else:
                out.append(f"{aa}{modstr}")

        return out

    @classmethod
    def from_proforma(
        cls,
        sequence: str,
    ) -> Peptide:
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
        pep, meta = proforma.parse(sequence)
        try:
            charge = meta["charge_state"].charge
        except AttributeError:
            charge = None

        static_mods = {}
        for mod_rule in meta["fixed_modifications"]:
            for res in mod_rule.targets:
                static_mods[res] = mod_rule.modification_tag

        seq = [None] * len(pep)
        mods = [None] * (len(pep) + 2)
        mods[0] = meta["n_term"]
        mods[-1] = meta["c_term"]
        for idx, (res, var_mods) in enumerate(pep):
            seq[idx] = res
            if res in static_mods:
                if var_mods is None:
                    var_mods = []

                var_mods.insert(0, static_mods[res])

            mods[idx + 1] = var_mods

        return cls("".join(seq), mods, charge)

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
class PeptideIons:
    """The ions generated by a peptide.

    Parameters
    ----------
    tokens : list[str]
        The string tokens that comprise the peptide sequence.
    precursor : float
        The monoisotopic m/z of the precursor ion.
    fragments : torch.Tensor[float]
        The generated fragment ions originated from the peptide.

    """

    tokens: list[str]
    precursor: float
    fragments: torch.Tensor[float]

    @property
    def sequence(self) -> str:
        """The peptide sequence."""
        return "".join(self.tokens)

    @property
    def b_ions(self) -> torch.Tensor[float]:
        """The b ion series."""
        return self.fragments[0, :, :]

    @property
    def y_ions(self) -> torch.Tensor[float]:
        """The y ion series."""
        return self.fragments[1, :, :]


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


class MassSpectrum(MsmsSpectrum):
    """A mass spectrum.

    Parameters
    ----------
    filename: str
        The file from which the spectrum originated.
    scan_id: str
        The Hupo PSI standard scan identifier.
    mz: array of shape (n_peaks,)
        The m/z values.
    intensity: array of shape (n_peaks, )
        The intensity values.
    retention_time: float, optional
        The measured retention time.
    ion_mobility: float, optional
        The measured ion mobility.
    precursor_mz: float, optional
        The precursor ion m/z, if applicable.
    precursor_charge: int, optional
        The precursor charge, if applicable.
    label: str, optional
        A label for the mass spectrum. This is typically an
        annotation, such as the generating peptide sequence,
        but is distinct from spectrum_utils' annotation.

    """

    def __init__(
        self,
        filename: str,
        scan_id: str,
        mz: ArrayLike,
        intensity: ArrayLike,
        ms_level: int = None,
        retention_time: float | None = None,
        ion_mobility: float | None = None,
        precursor_mz: float | None = None,
        precursor_charge: int | None = None,
        label: str | None = None,
    ) -> None:
        """Initialize a MassSpectrum."""
        self.filename = filename
        self.scan_id = scan_id
        self.ms_level = ms_level
        self.label = label

        # Not currently supported by spectrum_utils:
        self.ion_mobility = ion_mobility

        # spectrum_utils requires a precursor. Will remove after
        # I make a PR:
        if precursor_mz is None:
            precursor_mz = np.nan

        if precursor_charge is None:
            precursor_charge = np.nan

        super().__init__(
            identifier=str(scan_id),
            precursor_mz=precursor_mz,
            precursor_charge=precursor_charge,
            mz=mz,
            intensity=intensity,
            retention_time=retention_time,
        )

    @property
    def usi(self) -> str:
        """The Universal Spectrum Identifier."""
        index_type, scan = self.scan_id.split("=")
        fname = re.sub(
            ".(gz|tar|tar.gz|zip|bz2|tar.bz2)$",
            "",
            self.filename,
            flags=re.IGNORECASE,
        )
        fname = re.sub(
            ".(raw|mzml|mzxml|mgf|d|wiff)$",
            "",
            fname,
            flags=re.IGNORECASE,
        )
        return ":".join([fname, index_type, scan])

    @MsmsSpectrum.intensity.setter
    def intensity(self, values):
        """Set the intensity array."""
        self._inner._intensity = values

    @MsmsSpectrum.mz.setter
    def mz(self, values):
        """Set the m/z array."""
        self._inner._mz = values

    @property
    def precursor_mass(self) -> float:
        """The monoisotopic mass."""
        return (self.precursor_mz - PROTON) * self.precursor_charge

    def to_tensor(self) -> torch.tensor:
        """Combine the m/z and intensity arrays into a single tensor.

        Returns
        -------
        torch.tensor of shape (n_peaks, 2)
            The mass spectrum information.

        """
        return torch.tensor(np.vstack([self.mz, self.intensity]).T)


def _resolve_mod(
    mod: MassModification | GenericModification | str | float,
) -> MassModification | GenericModification:
    """Resolve the type of a modification.

    Parameters
    ----------
    mod : MassModification, GenericModification, str, or float
        The modification to resolve.

    Returns
    -------
    MassModification or GenericModification
        The best modification for the input type.

    """
    try:
        mod = mod.value
    except AttributeError:
        pass

    try:
        return MassModification(float(mod))
    except ValueError:
        return GenericModification(str(mod))
