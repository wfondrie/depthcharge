"""Test that our fundamental dataclasses work."""

import numpy as np
import pytest
import torch
from pyteomics.proforma import ProFormaError

from depthcharge.primitives import (
    MassSpectrum,
    Molecule,
    Peptide,
    PeptideIons,
)


def test_peptide_init():
    """Test our peptide dataclass."""
    peptide = "LESLIEK"
    assert Peptide(peptide).split() == list(peptide)
    assert Peptide(peptide, None, None).charge is None
    assert Peptide(peptide, None, 2).charge == 2

    mods = [-10.0, None, None, 71.0, None, None, "Phospho", None, +1]
    expected = [
        "[-10.000000]-",
        "L",
        "E",
        "S[+71.000000]",
        "L",
        "I",
        "E[Phospho]",
        "K",
        "-[+1.000000]",
    ]

    parsed = Peptide(peptide, mods)
    assert parsed.split() == expected

    mods = [None, None, None, 71.0, None, None, "Phospho", None, None]
    expected = [
        "L",
        "E",
        "S[+71.000000]",
        "L",
        "I",
        "E[Phospho]",
        "K",
    ]
    parsed = Peptide(peptide, mods)
    assert parsed.split() == expected


def test_almost_proforma():
    """Test a peptide lacking an explicit sign."""
    parsed = Peptide.from_proforma("LES[79.0]LIEK")
    assert parsed.split() == ["L", "E", "S[+79.000000]", "L", "I", "E", "K"]


def test_peptide_from_proforma():
    """Test proforma parsing."""
    parsed = Peptide.from_proforma("LESLIEK/2")
    assert parsed.split() == list("LESLIEK")
    assert parsed.charge == 2

    parsed = Peptide.from_proforma("<[Phospho]@C,T>CAT")
    assert parsed.split() == ["C[Phospho]", "A", "T[Phospho]"]

    parsed = Peptide.from_proforma("ED[Phospho]IT[Obs:+1]H")
    assert parsed.split() == ["E", "D[Phospho]", "I", "T[+1.000000]", "H"]

    # If multiple mods exist on a single AA, the mass should be summed.
    parsed = Peptide.from_proforma("<[Phospho]@C,T>C[+1]AT")
    assert parsed.split() == ["C[+80.966331]", "A", "T[Phospho]"]

    reparsed = Peptide.from_proforma(parsed.proforma)
    assert parsed.split() == reparsed.split()


def test_peptide_from_massivekb():
    """Test MassIVE-KB parsing."""
    parsed = Peptide.from_massivekb("LESLIEK")
    assert parsed.split() == list("LESLIEK")

    parsed = Peptide.from_massivekb("+42.011EDITH", 2)
    assert parsed.charge == 2
    assert parsed.split() == ["[Acetyl]-", "E", "D", "I", "T", "H"]

    with pytest.raises(ProFormaError):
        Peptide.from_massivekb("LES+79LIEK")


def test_peptide_ions():
    """Test peptide ions."""
    ions = PeptideIons(["A", "B"], 42.0, torch.tensor([[[0]], [[1]]]))
    assert ions.sequence == "AB"
    assert ions.b_ions == torch.tensor([[0]])
    assert ions.y_ions == torch.tensor([[1]])


def test_molecule_init():
    """Test that molecule initialize correctly."""
    etoh1 = Molecule("CCO")
    etoh2 = Molecule("C(O)C")
    assert etoh1.smiles == etoh2.smiles
    assert etoh1.to_selfies() == etoh2.to_selfies()

    shown = etoh1.show()
    assert shown is not None

    etoh1_selfie = etoh1.to_selfies()
    assert Molecule.from_selfies(etoh1_selfie) == etoh1

    with pytest.raises(ValueError):
        Molecule("not a smiles string")


def test_mass_spectrum():
    """Test that the mass spectrum is created correctly."""
    mzs = np.arange(10)
    intensities = np.arange(10) + 100

    spec = MassSpectrum("test", "scan=1", mzs, intensities)
    assert spec.usi == "test:scan:1"

    spec = MassSpectrum(
        "test",
        "scan=1",
        mzs,
        intensities,
        precursor_mz=10.0,
        precursor_charge=3,
    )
    assert spec.to_tensor().shape == (10, 2)

    new_mzs = np.ones_like(spec.mz)
    new_intensities = np.ones_like(spec.intensity)
    spec.mz = new_mzs
    spec.intensity = new_intensities

    np.testing.assert_equal(spec.mz, new_mzs)
    np.testing.assert_equal(spec.intensity, new_intensities)
