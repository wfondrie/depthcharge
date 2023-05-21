"""Test peptide tokenizers."""

import torch

from depthcharge.tokenizers.peptides import (
    PeptideTokenizer,
)

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
        ("S[Phospho]", 79.966331),
        ("E[-10.000000]", -10.0),
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


def test_mskb_init():
    """Test that the MassIVE-KB dataset works."""
    seqs = ["+42.011EDITH"]
    mskb = PeptideTokenizer.from_massivekb(False, False)
    tokens = mskb.tokenize(seqs, to_strings=True)[0]
    assert tokens == ["[Acetyl]-", "E", "D", "I", "T", "H"]


def test_precursor_mz():
    """Test calculation of precurosr m/z."""
    pass


def test_fragment_mz():
    """Test fragment calculations."""
    tokenizer = PeptideTokenizer()
    ions = tokenizer.fragment_mz(["LESLIEK"], 1)[0]
    expected = torch.tensor(LESLIEK_PLUS_ONE)[:, :, None]
    torch.testing.assert_close(ions, expected, check_dtype=False)

    ions = tokenizer.fragment_mz(["LESLIEK"], 2)[0]
    expected = torch.cat(
        [
            torch.tensor(LESLIEK_PLUS_ONE)[:, :, None],
            torch.tensor(LESLIEK_PLUS_TWO)[:, :, None],
        ],
        dim=2,
    )
    torch.testing.assert_close(ions, expected, check_dtype=False)

    ions = tokenizer.fragment_mz(["LESLIEK/1"], None)[0]
    expected = torch.tensor(LESLIEK_PLUS_ONE)[:, :, None]
    torch.testing.assert_close(ions, expected, check_dtype=False)

    ions = tokenizer.fragment_mz(["LESLIEK/3"], None)[0]
    expected = torch.cat(
        [
            torch.tensor(LESLIEK_PLUS_ONE)[:, :, None],
            torch.tensor(LESLIEK_PLUS_TWO)[:, :, None],
        ],
        dim=2,
    )
    torch.testing.assert_close(ions, expected, check_dtype=False)

    tokenizer = PeptideTokenizer.from_proforma(["[+10]-LESLIEK"])
    ions = tokenizer.fragment_mz(["[+10.000000]-LESLIEK"], 1)[0]
    expected = torch.tensor(LESLIEK_PLUS_ONE)[:, :, None]
    expected[0, :, :] += 10
    torch.testing.assert_close(ions, expected, check_dtype=False)
