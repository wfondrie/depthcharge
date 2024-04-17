"""Test the molecule tokenizer."""

import pytest

from depthcharge.tokenizers import MoleculeTokenizer


@pytest.mark.parametrize(
    ["mode", "vocab", "len_vocab"],
    [
        ("basic", None, 69),
        ("basic", ["x", "y"], 2),
        ("selfies", ["[C][O][C]", "[F][C][F]", "[O][=O]"], 4),
        ("selfies", "[C][O]", 2),
        ("smiles", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 8),
        ("smiles", ["CN", "CC(=O)O"], 5),
    ],
)
def test_init(mode, vocab, len_vocab):
    """Test initialization."""
    if mode == "smiles":
        tokenizer = MoleculeTokenizer.from_smiles(vocab)
    elif mode == "selfies":
        tokenizer = MoleculeTokenizer.from_selfies(vocab)
    else:
        tokenizer = MoleculeTokenizer(vocab)

    assert len(tokenizer.selfies_vocab) == len_vocab


@pytest.mark.parametrize(
    "molecule",
    [
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "[C][N][C][=N][C][=C][Ring1][Branch1][C][=Branch1][C][=O][N][Branch1]"
        "[C][C][C][=Branch1][C][=O][N][Ring1][=Branch2][C]",
    ],
)
def test_split(molecule):
    """Test that split works as expected."""
    expected = [
        "[C]",
        "[N]",
        "[C]",
        "[=N]",
        "[C]",
        "[=C]",
        "[Ring1]",
        "[Branch1]",
        "[C]",
        "[=Branch1]",
        "[C]",
        "[=O]",
        "[N]",
        "[Branch1]",
        "[C]",
        "[C]",
        "[C]",
        "[=Branch1]",
        "[C]",
        "[=O]",
        "[N]",
        "[Ring1]",
        "[=Branch2]",
        "[C]",
    ]

    tokenizer = MoleculeTokenizer()
    assert expected == tokenizer.split(molecule)
