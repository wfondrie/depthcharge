"""Test the datasets."""

import pickle
import shutil
import warnings
from collections.abc import Iterable

import polars as pl
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
    hash_peak_file,
)
from depthcharge.testing import assert_dicts_equal
from depthcharge.tokenizers import MoleculeTokenizer, PeptideTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    """Use a tokenizer for every test."""
    return PeptideTokenizer()


def _modified_copy(peak_file, new_file):
    """Copy a peak file, changing its contents but not its spectra."""
    new_file.parent.mkdir(parents=True, exist_ok=True)
    new_file.write_text(peak_file.read_text() + "\n")
    return new_file


def test_addition(mgf_small, tmp_path):
    """Testing adding a file."""
    dataset = SpectrumDataset(mgf_small, path=tmp_path / "test", batch_size=1)
    assert dataset.n_spectra == 2
    assert dataset.peak_file_hashes == [hash_peak_file(mgf_small)]

    # Adding the same file again is skipped:
    with pytest.warns(UserWarning, match="Skipped 1 peak file"):
        dataset = dataset.add_spectra(mgf_small)

    assert dataset.n_spectra == 2

    # Even if it is a copy in a different directory:
    copied = tmp_path / "other" / mgf_small.name
    copied.parent.mkdir()
    shutil.copy(mgf_small, copied)
    with pytest.warns(UserWarning, match="No new spectra"):
        dataset = dataset.add_spectra(copied)

    assert dataset.n_spectra == 2

    # A different file with the same name is added:
    modified = _modified_copy(mgf_small, tmp_path / "new" / mgf_small.name)
    dataset = dataset.add_spectra(modified)
    assert dataset.n_spectra == 4
    assert dataset.peak_files == [mgf_small.name]
    assert sorted(dataset.peak_file_hashes) == sorted(
        [hash_peak_file(mgf_small), hash_peak_file(modified)]
    )


def test_duplicate_inputs(mgf_small, tmp_path):
    """Test that the same peak file is only added once."""
    with pytest.warns(UserWarning, match=r"already added .*: small\.mgf$"):
        dataset = SpectrumDataset(
            [mgf_small, mgf_small], path=tmp_path / "test", batch_size=1
        )

    assert dataset.n_spectra == 2


def test_existing_path(mgf_small, tmp_path):
    """Test that an existing dataset is reused and extended."""
    path = tmp_path / "test.lance"
    other = _modified_copy(mgf_small, tmp_path / "other.mgf")

    SpectrumDataset(mgf_small, path=path, batch_size=1)

    # The same peak files are not parsed again:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dataset = SpectrumDataset(mgf_small, path=path, batch_size=1)

    assert dataset.n_spectra == 2

    # Only new peak files are added:
    dataset = SpectrumDataset([mgf_small, other], path=path, batch_size=1)
    assert dataset.n_spectra == 4
    assert sorted(dataset.peak_file_hashes) == sorted(
        [hash_peak_file(mgf_small), hash_peak_file(other)]
    )

    # Overwrite replaces the dataset:
    dataset = SpectrumDataset(other, path=path, batch_size=1, overwrite=True)
    assert dataset.n_spectra == 2
    assert dataset.peak_file_hashes == [hash_peak_file(other)]


def test_existing_path_errors(mgf_small, tmp_path):
    """Test inputs that cannot be added to an existing dataset."""
    path = tmp_path / "test.lance"
    df = arrow.spectra_to_df(mgf_small, progress=False)
    SpectrumDataset(df, path=path, batch_size=1)

    # DataFrames cannot be checked for duplicates:
    with pytest.raises(ValueError, match="cannot be checked"):
        SpectrumDataset(df, path=path, batch_size=1)

    # Custom fields must already be in the dataset:
    seq = CustomField("seq", lambda x: x["params"]["seq"], pa.string())
    with pytest.raises(ValueError, match="missing the custom fields: seq"):
        SpectrumDataset(
            mgf_small,
            path=path,
            batch_size=1,
            parse_kwargs={"custom_fields": seq},
        )

    # Datasets without peak file hashes cannot be extended:
    SpectrumDataset(
        df.drop("peak_file_hash"), path=path, batch_size=1, overwrite=True
    )
    with pytest.raises(ValueError, match="previous version"):
        SpectrumDataset(mgf_small, path=path, batch_size=1)

    with pytest.raises(ValueError, match=r"Unexpected: \['peak_file_hash'\]"):
        SpectrumDataset.from_lance(path, 1).add_spectra(mgf_small)


def test_add_to_dataframe_dataset(mgf_small, tmp_path):
    """Test adding peak files to a dataset created from a DataFrame."""
    other = _modified_copy(mgf_small, tmp_path / "other.mgf")
    df = arrow.spectra_to_df(mgf_small, progress=False)
    path = tmp_path / "test.lance"
    dataset = SpectrumDataset(df, path=path, batch_size=1)
    dataset.add_spectra(other)
    assert dataset.n_spectra == 4

    dataset = SpectrumDataset(
        [mgf_small, other, tmp_path / "other.mgf"], path=path, batch_size=1
    )
    assert dataset.n_spectra == 4


def test_indexing(tokenizer, mgf_small, tmp_path):
    """Test retrieving spectra."""
    mgf_small2 = _modified_copy(mgf_small, tmp_path / "mgf_small2.mgf")

    dataset = SpectrumDataset(
        [mgf_small, mgf_small2], path=tmp_path / "test", batch_size=1
    )

    assert dataset.path == tmp_path / "test.lance"

    spec = dataset[0]
    assert len(spec) == 8
    assert spec["peak_file_hash"] == [hash_peak_file(mgf_small)]
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
        overwrite=True,
    )
    spec = dataset[0]
    assert len(spec) == 9
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
    assert len(spec) == 9
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
    assert len(spec) == 9
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
            overwrite=True,
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


def test_pad_fields(tmp_path):
    """Test padding additional list columns."""
    spectra = pl.DataFrame(
        {
            "mz_array": [[1.0, 2.0], [3.0, 4.0, 5.0]],
            "intensity_array": [[10.0, 20.0], [30.0, 40.0, 50.0]],
            "custom_array": [[1.0, 2.0], [3.0, 4.0, 5.0]],
            "scalar": [1.0, 2.0],
            "nulls": [[1.0], None],
        }
    )
    expected = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]])

    # Not padded by default:
    dataset = SpectrumDataset(spectra, path=tmp_path / "test", batch_size=2)
    assert isinstance(next(iter(dataset))["custom_array"], list)

    # SpectrumDataset:
    dataset = SpectrumDataset(
        spectra,
        path=tmp_path / "test",
        batch_size=2,
        pad_fields="custom_array",
        overwrite=True,
    )
    batch = next(iter(dataset))
    torch.testing.assert_close(batch["custom_array"], expected)
    torch.testing.assert_close(batch["mz_array"], expected)

    # Reopened from lance:
    dataset = SpectrumDataset.from_lance(
        tmp_path / "test.lance", 2, pad_fields=["custom_array"]
    )
    torch.testing.assert_close(next(iter(dataset))["custom_array"], expected)

    # Indexing:
    torch.testing.assert_close(dataset[1]["custom_array"], expected[[1]])

    # Missing columns are ignored:
    dataset = SpectrumDataset.from_lance(
        tmp_path / "test.lance",
        2,
        pad_fields=["custom_array", "missing"],
        columns=["mz_array", "intensity_array"],
    )
    batch = next(iter(dataset))
    assert "custom_array" not in batch
    torch.testing.assert_close(batch["mz_array"], expected)

    # Streaming:
    dataset = StreamingSpectrumDataset(spectra, 2, pad_fields="custom_array")
    torch.testing.assert_close(next(iter(dataset))["custom_array"], expected)

    # Columns that can't be padded:
    for field in ["scalar", "nulls"]:
        dataset = StreamingSpectrumDataset(spectra, 2, pad_fields=field)
        with pytest.raises(ValueError, match=f"pad the '{field}' column"):
            next(iter(dataset))


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


def test_to_tensor_types(tmp_path):
    """Test that columns are converted to the expected types."""
    spectra = pl.DataFrame(
        {
            "mz_array": [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]],
            "intensity_array": [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]],
            "ints": pl.Series([1, 2, 3], dtype=pl.UInt8),
            "floats": [1.5, 2.5, 3.5],
            "bools": [True, False, True],
            "strings": ["a", "b", "c"],
            "nulls": [1.0, None, 3.0],
            "even": [[1, 2], [3, 4], [5, 6]],
            "ragged": [[1.0], [2.0, 3.0], []],
        }
    )
    expected = {
        "mz_array": torch.tensor(
            [[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [6.0, 0.0, 0.0]]
        ),
        "ints": torch.tensor([1, 2, 3]),
        "floats": torch.tensor([1.5, 2.5, 3.5]),
        "bools": torch.tensor([True, False, True]),
        "strings": ["a", "b", "c"],
        "nulls": [1.0, None, 3.0],
        "even": torch.tensor([[1, 2], [3, 4], [5, 6]]),
        "ragged": [[1.0], [2.0, 3.0], []],
    }

    def check(batch: dict, rows: list[int]) -> None:
        """Check a batch against the expected values."""
        torch.testing.assert_close(batch["intensity_array"], batch["mz_array"])
        for key, val in expected.items():
            if key == "ragged":
                assert len(batch[key]) == len(rows)
                for row, out in zip(rows, batch[key]):
                    torch.testing.assert_close(
                        out, torch.tensor(val[row], dtype=torch.float32)
                    )
            elif key == "mz_array":
                length = max(len(spectra["mz_array"][r]) for r in rows)
                torch.testing.assert_close(batch[key], val[rows][:, :length])
            elif isinstance(val, torch.Tensor):
                torch.testing.assert_close(batch[key], val[rows])
            else:
                assert batch[key] == [val[r] for r in rows]

    dataset = SpectrumDataset(spectra, path=tmp_path / "test", batch_size=3)
    check(next(iter(dataset)), [0, 1, 2])
    check(dataset[[1, 2]], [1, 2])
    check(dataset[[0, 1]], [0, 1])

    dataset = SpectrumDataset.from_lance(tmp_path / "test.lance", 3)
    check(next(iter(dataset)), [0, 1, 2])

    dataset = StreamingSpectrumDataset(spectra, 3)
    check(next(iter(dataset)), [0, 1, 2])

    # Later batches may start at an offset in the Arrow arrays:
    dataset = SpectrumDataset.from_lance(tmp_path / "test.lance", 2)
    batches = list(dataset)
    check(batches[0], [0, 1])
    torch.testing.assert_close(batches[1]["mz_array"], torch.tensor([[6.0]]))
    torch.testing.assert_close(batches[1]["even"], torch.tensor([[5, 6]]))
    torch.testing.assert_close(batches[1]["ragged"], torch.empty(1, 0))


@pytest.fixture
def ordered_spectra():
    """Spectra with an increasing scan_id."""
    return pl.DataFrame(
        {
            "scan_id": [f"scan={i}" for i in range(50)],
            "mz_array": [[float(i)] for i in range(50)],
            "intensity_array": [[1.0]] * 50,
            "seq": ["LESLIEK"] * 50,
        }
    )


def _scan_ids(dataset: Iterable) -> list[str]:
    """Get the scan IDs from each batch.

    Parameters
    ----------
    dataset : Iterable
        The dataset or DataLoader to iterate over.

    Returns
    -------
    list of str
        The scan IDs in the order they were loaded.

    """
    return [scan for batch in dataset for scan in batch["scan_id"]]


def test_shuffle(tmp_path, ordered_spectra, tokenizer):
    """Test shuffling spectra."""
    expected = ordered_spectra["scan_id"].to_list()
    path = tmp_path / "test.lance"

    # Not shuffled by default:
    dataset = SpectrumDataset(ordered_spectra, 8, path=path)
    assert _scan_ids(dataset) == expected
    assert _scan_ids(dataset) == expected

    for dataset in [
        SpectrumDataset.from_lance(path, 8, shuffle=True, seed=1),
        AnnotatedSpectrumDataset.from_lance(
            path, "seq", tokenizer, 8, shuffle=True, seed=1
        ),
    ]:
        epochs = [_scan_ids(dataset) for _ in range(3)]
        for epoch in epochs:
            assert epoch != expected
            assert sorted(epoch) == sorted(expected)

        assert epochs[0] != epochs[1] != epochs[2]

        # Batches are the requested size:
        sizes = [len(b["scan_id"]) for b in dataset]
        assert sizes == [8] * 6 + [2]

        # Setting the epoch reproduces the order:
        dataset.set_epoch(1)
        assert _scan_ids(dataset) == epochs[1]

    # The same seed gives the same order:
    dataset = SpectrumDataset.from_lance(path, 8, shuffle=True, seed=1)
    assert _scan_ids(dataset) == epochs[0]

    # The default seed comes from PyTorch:
    torch.manual_seed(42)
    first = _scan_ids(SpectrumDataset.from_lance(path, 8, shuffle=True))
    torch.manual_seed(42)
    second = _scan_ids(SpectrumDataset.from_lance(path, 8, shuffle=True))
    assert first == second

    # When creating a dataset:
    dataset = SpectrumDataset(
        ordered_spectra, 8, path=path, overwrite=True, shuffle=True, seed=1
    )
    assert _scan_ids(dataset) == epochs[0]

    # Incompatible options:
    with pytest.raises(ValueError, match="cannot be combined.*filter"):
        SpectrumDataset.from_lance(path, 8, shuffle=True, filter="seq = 'A'")

    # The LanceDataset options still work without shuffling:
    dataset = SpectrumDataset.from_lance(path, 8, filter="scan_id = 'scan=3'")
    assert _scan_ids(dataset) == ["scan=3"]


@pytest.mark.parametrize("shuffle", [False, True])
def test_distributed(tmp_path, ordered_spectra, monkeypatch, shuffle):
    """Test that each process loads different spectra."""
    dataset = SpectrumDataset(
        ordered_spectra, 8, path=tmp_path / "test", shuffle=shuffle, seed=1
    )

    loaded = []
    for rank in range(3):
        monkeypatch.setattr(
            "depthcharge.data.spectrum_datasets.get_global_rank",
            lambda rank=rank: rank,
        )
        monkeypatch.setattr(
            "depthcharge.data.spectrum_datasets.get_global_world_size",
            lambda: 3,
        )
        dataset.set_epoch(0)
        loaded.append(_scan_ids(dataset))

    assert all(loaded)
    assert sorted(sum(loaded, [])) == sorted(ordered_spectra["scan_id"])


def test_dataloader_workers(tmp_path, ordered_spectra):
    """Test that DataLoader workers load each spectrum once."""
    dataset = SpectrumDataset(
        ordered_spectra, 8, path=tmp_path / "test", shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    scans = _scan_ids(loader)
    assert scans != ordered_spectra["scan_id"].to_list()
    assert sorted(scans) == sorted(ordered_spectra["scan_id"])


def test_streaming_batch_size(tmp_path):
    """Test that DataFrame and parquet inputs respect the batch size."""
    spectra = pl.concat(
        [
            pl.DataFrame(
                {
                    "scan_id": [f"scan={i}"],
                    "mz_array": [[1.0, 2.0]],
                    "intensity_array": [[3.0, 4.0]],
                }
            )
            for i in range(5)
        ],
        rechunk=False,
    )
    parquet_file = tmp_path / "spectra.parquet"
    spectra.write_parquet(parquet_file)

    for data in [spectra, spectra.lazy(), parquet_file]:
        batches = list(StreamingSpectrumDataset(data, batch_size=2))
        assert [len(b["scan_id"]) for b in batches] == [2, 2, 1]
        assert [s for b in batches for s in b["scan_id"]] == [
            f"scan={i}" for i in range(5)
        ]
