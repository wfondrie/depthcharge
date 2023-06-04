"""Test peptide tokenizers."""
import math
from functools import partial

import pytest
import torch
from pyteomics import mass

from depthcharge.tokenizers.peptides import PeptideTokenizer

# Calculated using Pyteomics:
# These are [b_ions, y_ions]
LESLIEK_PLUS_ONE = [
    [
        114.09134044390001,
        243.13393353187,
        330.16596193614004,
        443.25002591327006,
        556.3340898904,
        685.37668297837,
    ],
    [
        147.11280416447,
        276.15539725243997,
        389.23946122957,
        502.3235252067,
        589.3555536109699,
        718.39814669894,
    ],
]

LESLIEK_PLUS_TWO = [
    [
        57.54930845533501,
        122.07060499932,
        165.58661920145502,
        222.12865119002004,
        278.670683178585,
        343.19197972257,
    ],
    [
        74.06004031561999,
        138.581336859605,
        195.12336884817,
        251.66540083673502,
        295.18141503886994,
        359.702711582855,
    ],
]


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


def test_precursor_ions():
    """Test calculation of precurosr m/z."""
    tokenizer = PeptideTokenizer()

    aa_mass = dict(mass.std_aa_mass)
    aa_mass["a"] = 42.010565
    aa_mass["o"] = 15.994915
    pymass = partial(mass.fast_mass, ion_type="M", aa_mass=aa_mass)
    close = partial(math.isclose, rel_tol=1e-6)

    seq = "LESLIEK"
    assert close(tokenizer.ions(seq, 1)[0].precursor, pymass(seq, charge=1))
    assert close(tokenizer.ions(seq, 2)[0].precursor, pymass(seq, charge=2))
    assert close(tokenizer.ions(seq, 3)[0].precursor, pymass(seq, charge=3))

    seq = "[Acetyl]-LESLIM[Oxidation]K"
    with pytest.raises(ValueError):
        tokenizer.ions(seq, 1)

    tokenizer = PeptideTokenizer.from_proforma([seq])
    seq2 = "aLESLIMoK"
    assert close(tokenizer.ions(seq, 1)[0].precursor, pymass(seq2, charge=1))
    assert close(tokenizer.ions(seq, 2)[0].precursor, pymass(seq2, charge=2))
    assert close(tokenizer.ions(seq, 3)[0].precursor, pymass(seq2, charge=3))


def test_fragment_ions():
    """Test ion calculations."""
    tokenizer = PeptideTokenizer()
    ions = tokenizer.ions(["LESLIEK"], [1])[0]
    expected = torch.tensor(LESLIEK_PLUS_ONE)[:, :, None]
    torch.testing.assert_close(ions.fragments, expected, check_dtype=False)

    ions = tokenizer.ions(["LESLIEK"], [2])[0]
    expected = torch.cat(
        [
            torch.tensor(LESLIEK_PLUS_ONE)[:, :, None],
            torch.tensor(LESLIEK_PLUS_TWO)[:, :, None],
        ],
        dim=2,
    )
    torch.testing.assert_close(ions.fragments, expected, check_dtype=False)

    ions = tokenizer.ions(["LESLIEK/1"], None)[0]
    expected = torch.tensor(LESLIEK_PLUS_ONE)[:, :, None]
    torch.testing.assert_close(ions.fragments, expected, check_dtype=False)

    ions = tokenizer.ions(["LESLIEK/3"], None)[0]
    expected = torch.cat(
        [
            torch.tensor(LESLIEK_PLUS_ONE)[:, :, None],
            torch.tensor(LESLIEK_PLUS_TWO)[:, :, None],
        ],
        dim=2,
    )
    torch.testing.assert_close(ions.fragments, expected, check_dtype=False)

    tokenizer = PeptideTokenizer.from_proforma(["[+10]-LESLIEK"])
    ions = tokenizer.ions(["[+10.000000]-LESLIEK"], 1)[0]
    expected = torch.tensor(LESLIEK_PLUS_ONE)[:, :, None]
    expected[0, :, :] += 10
    torch.testing.assert_close(ions.fragments, expected, check_dtype=False)
