"""Test that our fundamental dataclasses work."""
import pytest
from lark.exceptions import UnexpectedCharacters

from depthcharge.base.dataclasses import (
    Peptide,
)


def test_peptide_init():
    """Test our peptide dataclass."""
    peptide = "LESLIEK"
    assert Peptide(peptide).split() == list(peptide)
    assert Peptide(peptide, None, None).charge is None
    assert Peptide(peptide, None, 2).charge == 2

    mods = [-10.0, None, None, 71.0, None, None, "blah", None, "c"]
    expected = [
        "[-10.000000]-",
        "L",
        "E",
        "S[+71.000000]",
        "L",
        "I",
        "E[blah]",
        "K",
        "-[c]",
    ]

    parsed = Peptide(peptide, mods)
    assert parsed.split() == expected

    mods = [None, None, None, 71.0, None, None, "blah", None, None]
    expected = [
        "L",
        "E",
        "S[+71.000000]",
        "L",
        "I",
        "E[blah]",
        "K",
    ]
    parsed = Peptide(peptide, mods)
    assert parsed.split() == expected


def test_peptide_from_proforma():
    """Test proforma parsing."""
    parsed = Peptide.from_proforma("LESLIEK/2")
    assert parsed.split() == list("LESLIEK")
    assert parsed.charge == 2

    parsed = Peptide.from_proforma("<[Phospho]@C,T>CAT")
    assert parsed.split() == ["C[Phospho]", "A", "T[Phospho]"]

    parsed = Peptide.from_proforma("ED[Phospho]IT[Obs:+1]H")
    assert parsed.split() == ["E", "D[Phospho]", "I", "T[+1.000000]", "H"]


def test_peptide_from_massivekb():
    """Test MassIVE-KB parsing."""
    parsed = Peptide.from_massivekb("LESLIEK")
    assert parsed.split() == list("LESLIEK")

    parsed = Peptide.from_massivekb("+42.011EDITH", 2)
    assert parsed.charge == 2
    assert parsed.split() == ["[Acetyl]-", "E", "D", "I", "T", "H"]

    with pytest.raises(UnexpectedCharacters):
        Peptide.from_massivekb("LES+79LIEK")


def test_molecule_init():
    """Test that molecule initialize correctly."""
    pass
