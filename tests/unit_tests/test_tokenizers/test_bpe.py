"""Test byte-pair encoding (BPE) for tokenizers."""

import pickle
import random

import pyarrow as pa
import pytest
import torch

from depthcharge.data import (
    AnalyteDataset,
    AnnotatedSpectrumDataset,
    CustomField,
)
from depthcharge.tokenizers import MoleculeTokenizer, PeptideTokenizer
from depthcharge.transformers import AnalyteTransformerEncoder

PEPTIDES = [
    "LESLIEK",
    "PEPTIDEK",
    "LESM[Oxidation]EK",
    "GGLLEK",
    "LLLLK",
    "[Acetyl]-EDITHR",
]


def test_train_bpe():
    """Test learning merges on a toy corpus."""
    tokenizer = PeptideTokenizer()
    bpe = tokenizer.train_bpe(["LLEK", "LLEK", "LLAK"], len(tokenizer) + 2)

    assert bpe.merges == [("L", "L"), ("E", "K")]
    assert len(bpe) == len(tokenizer) + 2
    assert bpe.tokenize("LLEKLLA", to_strings=True) == [
        ["LL", "EK", "LL", "A"]
    ]

    # The original tokenizer is unchanged:
    assert tokenizer.merges == []
    assert tokenizer.tokenize("LLEK", to_strings=True) == [list("LLEK")]

    # Base token ids are unchanged:
    for token, idx in tokenizer.index.items():
        assert bpe.index[token] == idx

    # Training continues from existing merges:
    more = bpe.train_bpe(["LLEK", "LLEK"], len(bpe) + 1)
    assert more.merges == [("L", "L"), ("E", "K"), ("LL", "EK")]


def test_train_bpe_stopping():
    """Test the stopping criteria for learning merges."""
    seqs = ["LLEK", "LLEK", "LLAK"]
    tokenizer = PeptideTokenizer()

    # Only LL and EK occur at least twice:
    bpe = tokenizer.train_bpe(seqs, 1000, min_frequency=2)
    assert bpe.merges == [("L", "L"), ("E", "K"), ("LL", "EK")]

    bpe = tokenizer.train_bpe(seqs, 1000, min_frequency=3)
    assert bpe.merges == [("L", "L")]

    bpe = tokenizer.train_bpe(seqs, 1000, max_token_length=2)
    assert bpe.merges == [("L", "L"), ("E", "K")]

    bpe = tokenizer.train_bpe(seqs, 1000, min_frequency=1)
    assert bpe.tokenize(seqs, to_strings=True) == [
        ["LLEK"],
        ["LLEK"],
        ["LLAK"],
    ]


def test_merges_respect_modifications():
    """Test that merges never divide a modified residue."""
    seqs = ["LESM[Oxidation]EK"] * 3
    tokenizer = PeptideTokenizer.from_proforma(seqs)
    bpe = tokenizer.train_bpe(seqs, 1000)

    assert bpe.tokenize(seqs[0], to_strings=True) == [["LESM[Oxidation]EK"]]
    assert bpe.expansions[-1] == ("L", "E", "S", "M[Oxidation]", "E", "K")


@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("start_token", [None, "?"])
def test_round_trip(reverse, start_token):
    """Test that tokens are converted back to the same peptides."""
    tokenizer = PeptideTokenizer.from_proforma(
        PEPTIDES,
        reverse=reverse,
        start_token=start_token,
    )
    bpe = tokenizer.train_bpe(PEPTIDES * 2, len(tokenizer) + 20)
    assert len(bpe) > len(tokenizer)

    tokens = bpe.tokenize(PEPTIDES, add_start=True, add_stop=True)
    assert tokens.shape[0] == len(PEPTIDES)
    assert tokens.shape[1] < tokenizer.tokenize(PEPTIDES).shape[1]
    assert bpe.detokenize(tokens) == PEPTIDES

    # Merged tokens are expanded into residues:
    expected = [tokenizer.split(p) for p in PEPTIDES]
    if reverse:
        expected = [e[::-1] for e in expected]

    assert bpe.detokenize(tokens, join=False) == expected

    # Merged tokens are kept, but still read N- to C-terminus:
    unexpanded = bpe.detokenize(tokens, join=False, expand=False)
    assert ["".join(t) for t in unexpanded] == PEPTIDES
    assert sum(len(t) for t in unexpanded) < sum(len(e) for e in expected)


def test_precursor_ions():
    """Test that merged tokens have the masses of their residues."""
    tokenizer = PeptideTokenizer.from_proforma(PEPTIDES)
    bpe = tokenizer.train_bpe(PEPTIDES * 2, len(tokenizer) + 20)
    assert bpe.masses.shape == (len(bpe) + 1,)

    charges = torch.tensor([2, 3, 2, 1, 2, 3])
    torch.testing.assert_close(
        bpe.calculate_precursor_ions(PEPTIDES, charges),
        tokenizer.calculate_precursor_ions(PEPTIDES, charges),
    )


def test_molecules():
    """Test BPE with SMILES and SELFIES."""
    smiles = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O"]
    tokenizer = MoleculeTokenizer.from_smiles(smiles)
    bpe = tokenizer.train_bpe(smiles, len(tokenizer) + 5, min_frequency=1)
    assert len(bpe) == len(tokenizer) + 5

    selfies = [tokenizer.detokenize(tokenizer.tokenize(s))[0] for s in smiles]
    tokens = bpe.tokenize(smiles)
    assert tokens.shape[1] < tokenizer.tokenize(smiles).shape[1]
    assert bpe.detokenize(tokens) == selfies

    # A merged token contains whole SELFIES symbols:
    for expansion in bpe.expansions[len(tokenizer) + 1 :]:
        assert len(expansion) > 1
        assert all(e in tokenizer.index for e in expansion)

    # Merges can be passed to the class methods:
    assert MoleculeTokenizer.from_smiles(smiles, merges=bpe.merges).index == (
        bpe.index
    )


def test_dropout():
    """Test BPE-dropout."""
    tokenizer = PeptideTokenizer.from_proforma(PEPTIDES)
    bpe = tokenizer.train_bpe(PEPTIDES * 2, len(tokenizer) + 20)
    merged = bpe.tokenize(PEPTIDES)
    assert len(bpe._merge_cache) == len(PEPTIDES)

    # All merges are skipped:
    bpe._merge_cache.clear()
    torch.testing.assert_close(
        bpe.tokenize(PEPTIDES, bpe_dropout=1.0),
        tokenizer.tokenize(PEPTIDES),
    )
    assert not bpe._merge_cache

    # The attribute is used by default:
    bpe.bpe_dropout = 1.0
    torch.testing.assert_close(
        bpe.tokenize(PEPTIDES),
        tokenizer.tokenize(PEPTIDES),
    )
    torch.testing.assert_close(bpe.tokenize(PEPTIDES, bpe_dropout=0), merged)

    # Some merges are skipped, but the peptides are unchanged:
    bpe.bpe_dropout = 0.5
    random.seed(1)
    samples = [bpe.tokenize(PEPTIDES, to_strings=True) for _ in range(10)]
    assert len({str(s) for s in samples}) > 1
    for sample in samples:
        assert ["".join(s) for s in sample] == PEPTIDES


def test_save_and_load(tmp_path):
    """Test saving and loading merges."""
    tokenizer = PeptideTokenizer.from_proforma(PEPTIDES)
    bpe = tokenizer.train_bpe(PEPTIDES * 2, len(tokenizer) + 20)

    path = tmp_path / "merges.json"
    bpe.save_merges(path)
    loaded = tokenizer.load_merges(path)
    assert loaded.merges == bpe.merges
    assert loaded.index == bpe.index
    torch.testing.assert_close(loaded.masses, bpe.masses)
    torch.testing.assert_close(
        loaded.tokenize(PEPTIDES), bpe.tokenize(PEPTIDES)
    )

    # Merges can be passed to __init__ too:
    init = PeptideTokenizer.from_proforma(PEPTIDES, merges=bpe.merges)
    assert init.index == bpe.index

    with pytest.raises(ValueError, match="different vocabulary"):
        PeptideTokenizer().load_merges(path)


def test_pickle():
    """Test that a tokenizer with merges can be pickled."""
    tokenizer = PeptideTokenizer.from_proforma(PEPTIDES)
    bpe = tokenizer.train_bpe(PEPTIDES * 2, len(tokenizer) + 20)
    loaded = pickle.loads(pickle.dumps(bpe))
    torch.testing.assert_close(
        loaded.tokenize(PEPTIDES), bpe.tokenize(PEPTIDES)
    )


def test_invalid_merges():
    """Test that invalid merges raise errors."""
    with pytest.raises(ValueError, match="Unrecognized token in merge: X"):
        PeptideTokenizer(merges=[("X", "L")])

    with pytest.raises(ValueError, match="special token"):
        PeptideTokenizer(merges=[("K", "$")])

    with pytest.raises(ValueError, match="conflicts with an existing token"):
        MoleculeTokenizer(["x", "y", "xy"], merges=[("x", "y")])

    with pytest.raises(ValueError, match="Unrecognized token"):
        PeptideTokenizer().train_bpe(["PEPTIDEX"], 100)

    # Different merges can create the same token:
    tokenizer = PeptideTokenizer(
        merges=[("L", "K"), ("L", "LK"), ("L", "L"), ("LL", "K")]
    )
    assert len(tokenizer) == len(PeptideTokenizer()) + 3
    assert len(tokenizer.merges) == 4
    assert tokenizer.tokenize("LLK", to_strings=True) == [["LLK"]]


def test_conflicting_merge_is_skipped():
    """Test that training skips merges that conflict with a token."""
    tokenizer = MoleculeTokenizer(["x", "y", "xy"])
    bpe = tokenizer.train_bpe(["xyz", "xy"] * 2, 100, min_frequency=1)
    assert ("x", "y") not in bpe.merges


def test_datasets(mgf_small, tmp_path):
    """Test that datasets and models work with merges."""
    seqs = ["LESLIEK", "EDITHR"]
    tokenizer = PeptideTokenizer.from_proforma(seqs)
    bpe = tokenizer.train_bpe(seqs, len(tokenizer) + 3, min_frequency=1)

    dataset = AnnotatedSpectrumDataset(
        mgf_small,
        "seq",
        bpe,
        batch_size=2,
        path=tmp_path / "test.lance",
        parse_kwargs=dict(
            preprocessing_fn=[],
            custom_fields=CustomField(
                "seq", lambda x: x["params"]["seq"], pa.string()
            ),
        ),
    )
    batch = next(iter(dataset))
    torch.testing.assert_close(batch["seq"], bpe.tokenize(seqs, add_stop=True))
    assert bpe.detokenize(batch["seq"]) == seqs

    dset = AnalyteDataset(bpe, seqs)
    torch.testing.assert_close(dset.tokens, bpe.tokenize(seqs))

    model = AnalyteTransformerEncoder(bpe, 8, 2, 12)
    emb, _ = model(dset.tokens)
    assert emb.shape == (2, dset.tokens.shape[1] + 1, 8)
