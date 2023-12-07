"""Test peptide tokenizers."""

import torch
from pyteomics import mass

from depthcharge.tokenizers.peptides import PeptideTokenizer


def test_proforma_init():
    """Test initialization."""
    seqs = ["[+5.]-LES[Phospho]LIE[-10.0]K"]
    expected_tokens = [
        ("S[Phospho]", 87.032028435 + 79.966331),
        ("E[-10.000000]", 129.042593135 - 10.0),
        ("[+5.000000]-", 5.0),
    ]

    proforma = PeptideTokenizer.from_proforma(
        sequences=seqs,
        replace_isoleucine_with_leucine=True,
        reverse=False,
    )

    for key, val in expected_tokens:
        if isinstance(proforma, PeptideTokenizer):
            assert key in proforma.residues
        else:
            assert proforma.residues[key] == val

    tokens = proforma.tokenize(seqs, to_strings=True)[0]
    expected = [
        "[+5.000000]-",
        "L",
        "E",
        "S[Phospho]",
        "L",
        "L",
        "E[-10.000000]",
        "K",
    ]

    assert tokens == expected

    proforma = PeptideTokenizer.from_proforma(
        sequences=seqs,
        replace_isoleucine_with_leucine=False,
        reverse=False,
    )
    tokens = proforma.tokenize(["LESLIEK"], to_strings=True)[0]
    assert tokens == list("LESLIEK")
    tokens = proforma.tokenize(["LESLIEK"])
    orig = proforma.detokenize(tokens)
    assert orig == ["LESLIEK"]

    proforma = PeptideTokenizer.from_proforma(
        sequences=seqs,
        replace_isoleucine_with_leucine=False,
        reverse=True,
    )
    tokens = proforma.tokenize(["LESLIEK"], to_strings=True)[0]
    assert tokens == list("KEILSEL")
    tokens = proforma.tokenize(["LESLIEK"])
    orig = proforma.detokenize(tokens)
    assert orig == ["KEILSEL"]

    tokens = proforma.tokenize("LESLIEK", True, True)[0]
    assert "".join(tokens) == "KEILSEL$"


def test_mskb_init():
    """Test that the MassIVE-KB dataset works."""
    seqs = ["+42.011EDITH"]
    mskb = PeptideTokenizer.from_massivekb(False, False)
    tokens = mskb.tokenize(seqs, to_strings=True)[0]
    assert tokens == ["[Acetyl]-", "E", "D", "I", "T", "H"]


def test_torch_precursor_ions():
    """Test the calculation of the precursor m/z."""
    seqs = ["LESLIEK", "EDITHR"]
    charges = torch.tensor([2, 3])
    tokenizer = PeptideTokenizer.from_proforma(seqs)
    expected = torch.tensor(
        [
            mass.fast_mass(s, charge=z, ion_type="M")
            for s, z in zip(seqs, charges)
        ]
    )

    ions = tokenizer.calculate_precursor_ions(seqs, charges)
    torch.testing.assert_close(ions, expected)

    tokens = tokenizer.tokenize(seqs)
    ions = tokenizer.calculate_precursor_ions(tokens, charges)
    torch.testing.assert_close(ions, expected)


def test_single_peptide():
    """Test proforma from a single peptide."""
    tokenizer = PeptideTokenizer.from_proforma("[+10]-EDITHR")
    out = tokenizer.tokenize("LESLIEK")
    assert out.shape == (1, 7)
