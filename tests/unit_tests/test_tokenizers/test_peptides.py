"""Test peptide tokenizers."""

import pytest
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
    assert orig == ["LESLIEK"]

    tokens = proforma.tokenize("LESLIEK", True, True, True)[0]
    assert "".join(tokens) == "KEILSEL$"

    # Test a non-canonical AA:
    with pytest.raises(KeyError):
        PeptideTokenizer.from_proforma("TOBIN")


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

    ion = tokenizer.calculate_precursor_ions("LESLIEK", 2)
    expected = mass.fast_mass("LESLIEK", charge=2, ion_type="M")
    torch.testing.assert_close(ion, torch.tensor([expected]))


@pytest.mark.parametrize(("reverse",), [(True,), (False,)])
def test_reverse(reverse):
    """Test that reverse option correctly handles multi-character tokens."""
    tokenizer = PeptideTokenizer.from_proforma("[+10]-EDITHR", reverse=reverse)
    tokens = tokenizer.tokenize("[+10.000000]-PEPTLLDEK")
    seq = tokenizer.detokenize(tokens)[0]
    assert seq == "[+10.000000]-PEPTLLDEK"


def test_almost_compliant_proform():
    """Test initializing with a peptide without an expicit mass sign."""
    tokenizer = PeptideTokenizer.from_proforma("[10]-EDITHR")
    assert "[+10.000000]-" in tokenizer.residues


@pytest.mark.parametrize(
    ("start", "stop", "expected"),
    [
        (True, True, "ACD"),
        (True, False, "ACD$E"),
        (False, True, "?ACD"),
        (False, False, "?ACD$E"),
    ],
)
def test_trim(start, stop, expected):
    """Test that the start and stop tokens can be trimmed."""
    tokenizer = PeptideTokenizer(start_token="?")
    tokens = torch.tensor([[0, 2, 3, 4, 5, 1, 6]])
    out = tokenizer.detokenize(
        tokens,
        trim_start_token=start,
        trim_stop_token=stop,
    )

    assert out[0] == expected
