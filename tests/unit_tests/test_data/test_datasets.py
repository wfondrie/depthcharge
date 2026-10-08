"""Test the datasets."""

import pickle
import shutil

import pyarrow as pa
import pytest
import torch

from depthcharge.data import (
    AnalyteDataset,
    AnnotatedSpectrumDataset,
    CustomField,
    SpectrumDataset,
    StreamingSpectrumDataset,
    arrow,
)
from depthcharge.testing import assert_dicts_equal
from depthcharge.tokenizers import MoleculeTokenizer, PeptideTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    """Use a tokenizer for every test."""
    return PeptideTokenizer()


def test_addition(mgf_small, tmp_path):
    """Testing adding a file."""
    dataset = SpectrumDataset(mgf_small, path=tmp_path / "test", batch_size=1)
    assert dataset.n_spectra == 2

    dataset = dataset.add_spectra(mgf_small)
    assert dataset.n_spectra == 4


def test_indexing(tokenizer, mgf_small, tmp_path):
    """Test retrieving spectra."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)

    dataset = SpectrumDataset(
        [mgf_small, mgf_small2], path=tmp_path / "test", batch_size=1
    )

    assert dataset.path == tmp_path / "test.lance"

    spec = dataset[0]
    assert len(spec) == 7
    assert spec["peak_file"] == ["small.mgf"]
    assert spec["scan_id"] == ["index=0"]
    assert spec["ms_level"].item() == 2
    assert (spec["precursor_mz"].item() - 416.2448) < 0.001

    parse_kwargs = dict(
        preprocessing_fn=[],
        custom_fields=CustomField(
            "seq", lambda x: x["params"]["seq"], pa.string()
        ),
    )

    dataset = AnnotatedSpectrumDataset(
        [mgf_small, mgf_small2],
        "seq",
        tokenizer,
        path=tmp_path / "test.lance",
        batch_size=1,
        parse_kwargs=parse_kwargs,
    )
    spec = dataset[0]
    assert len(spec) == 8
    assert spec["mz_array"].shape == (
        1,
        14,
    )
    torch.testing.assert_close(
        spec["seq"], tokenizer.tokenize(["LESLIEK"], add_stop=True)
    )

    spec2 = dataset[3]
    assert spec2["mz_array"].shape == (
        1,
        24,
    )
    torch.testing.assert_close(
        spec2["seq"], tokenizer.tokenize(["EDITHR"], add_stop=True)
    )


def test_load(tokenizer, tmp_path, mgf_small):
    """Test saving and loading a dataset."""
    db_path = tmp_path / "test.lance"

    AnnotatedSpectrumDataset(
        mgf_small,
        "seq",
        tokenizer,
        1,
        db_path,
        parse_kwargs=dict(
            preprocessing_fn=[],
            custom_fields=CustomField(
                "seq", lambda x: x["params"]["seq"], pa.string()
            ),
        ),
    )

    dataset = AnnotatedSpectrumDataset.from_lance(db_path, "seq", tokenizer, 1)

    spec = dataset[0]
    assert len(spec) == 8
    assert spec["mz_array"].shape == (1, 14)
    torch.testing.assert_close(
        spec["seq"], tokenizer.tokenize(["LESLIEK"], add_stop=True)
    )

    spec2 = dataset[1]
    assert spec2["mz_array"].shape == (1, 24)
    torch.testing.assert_close(
        spec2["seq"], tokenizer.tokenize(["EDITHR"], add_stop=True)
    )

    dataset = SpectrumDataset.from_lance(db_path, 1)
    spec = dataset[0]
    assert len(spec) == 8
    assert spec["peak_file"] == ["small.mgf"]
    assert spec["scan_id"] == ["index=0"]
    assert spec["ms_level"] == 2
    assert (spec["precursor_mz"] - 416.2448) < 0.001


def test_formats(tmp_path, real_mgf, real_mzml, real_mzxml):
    """Test all of the supported formats."""
    df = arrow.spectra_to_df(real_mgf)
    parquet = arrow.spectra_to_parquet(
        real_mgf,
        parquet_file=tmp_path / "test.parquet",
    )

    data = [df, real_mgf, real_mzml, real_mzxml, parquet]
    for input_type in data:
        SpectrumDataset(
            spectra=input_type,
            path=tmp_path / "test",
            batch_size=1,
        )


def test_streaming_spectra(mgf_small):
    """Test the streaming dataset."""
    streamer = StreamingSpectrumDataset(mgf_small, batch_size=1)
    spec = next(iter(streamer))
    expected = SpectrumDataset(mgf_small, batch_size=1)[0]
    assert_dicts_equal(spec, expected)

    streamer = StreamingSpectrumDataset(mgf_small, batch_size=2)
    spec = next(iter(streamer))
    expected = SpectrumDataset(mgf_small, batch_size=1)[[0, 1]]
    assert_dicts_equal(spec, expected)


def test_analyte_dataset(tokenizer):
    """Test the peptide dataset."""
    seqs = ["LESLIEK", "EDITHR"]
    charges = torch.tensor([2, 3])
    dset = AnalyteDataset(tokenizer, seqs)
    torch.testing.assert_close(dset[0][0], tokenizer.tokenize("LESLIEK")[0])
    torch.testing.assert_close(dset[1][0][:6], tokenizer.tokenize("EDITHR")[0])
    assert len(dset) == 2

    seqs = ["LESLIEK", "EDITHR"]
    charges = torch.tensor([2, 3])
    target = torch.tensor([1.1, 2.2])
    other = torch.tensor([[1, 1], [2, 2]])
    dset = AnalyteDataset(tokenizer, seqs, charges, target, other)
    torch.testing.assert_close(dset[0][0], tokenizer.tokenize("LESLIEK")[0])
    torch.testing.assert_close(dset[1][0][:6], tokenizer.tokenize("EDITHR")[0])
    assert dset[0][1].item() == 2
    assert dset[1][1].item() == 3
    torch.testing.assert_close(dset[0][2], torch.tensor(1.1))
    torch.testing.assert_close(dset[1][3], other[1, :])
    assert len(dset) == 2

    torch.testing.assert_close(dset.tokens, tokenizer.tokenize(seqs))
    torch.testing.assert_close(dset.tensors[1], charges)


def test_with_molecule_tokenizer():
    """Test analyte dataset with a molecule tokenizer."""
    tokenizer = MoleculeTokenizer()
    smiles = ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC=CC(=O)C1=C(CCCC1(C)C)C"]
    tokens = tokenizer.tokenize(smiles)
    dset = AnalyteDataset(tokenizer, smiles)

    torch.testing.assert_close(dset.tokens, tokens)


def test_pickle(tokenizer, tmp_path, mgf_small):
    """Test that datasets can be pickled."""
    dataset = SpectrumDataset(mgf_small, batch_size=1, path=tmp_path / "test")
    pkl_file = tmp_path / "test.pkl"
    with pkl_file.open("wb+") as pkl:
        pickle.dump(dataset, pkl)

    with pkl_file.open("rb") as pkl:
        loaded = pickle.load(pkl)

    assert dataset.n_spectra == loaded.n_spectra

    dataset = AnnotatedSpectrumDataset(
        [mgf_small],
        tokenizer,
        "seq",
        batch_size=1,
        path=tmp_path / "test.lance",
    )
    pkl_file = tmp_path / "test.pkl"

    with pkl_file.open("wb+") as pkl:
        pickle.dump(dataset, pkl)

    with pkl_file.open("rb") as pkl:
        loaded = pickle.load(pkl)

    assert dataset.n_spectra == loaded.n_spectra
