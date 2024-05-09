"""Test PyTorch DataLoaders."""

import pyarrow as pa
import pytest
import torch
from torch.utils.data import DataLoader

from depthcharge.data import (
    AnalyteDataset,
    AnnotatedSpectrumDataset,
    CustomField,
    SpectrumDataset,
    StreamingSpectrumDataset,
)
from depthcharge.testing import assert_dicts_equal
from depthcharge.tokenizers import PeptideTokenizer


def test_spectrum_loader(mgf_small, tmp_path):
    """Test a normal spectrum dataset."""
    dset = SpectrumDataset(mgf_small, batch_size=2, path=tmp_path / "test")
    loader = DataLoader(dset)
    batch = next(iter(loader))
    assert len(batch) == 7
    assert batch["mz_array"].shape == (1, 2, 20)
    assert isinstance(batch["mz_array"], torch.Tensor)


@pytest.mark.parametrize(["samples", "batches"], [(1, 1), (50, 2), (4, 2)])
def test_spectrum_loader_samples(mgf_small, tmp_path, samples, batches):
    """Test sampling."""
    dset = SpectrumDataset(
        mgf_small,
        batch_size=1,
        path=tmp_path / "test",
        samples=samples,
    )

    loaded = list(DataLoader(dset))
    assert dset.n_spectra == 2
    assert len(loaded) == batches


def test_streaming_spectrum_loader(mgf_small, tmp_path):
    """Test streaming spectra."""
    streamer = StreamingSpectrumDataset(mgf_small, batch_size=2)
    loader = DataLoader(streamer)
    stream_batch = next(iter(loader))

    dset = SpectrumDataset(mgf_small, batch_size=2, path=tmp_path / "test")
    loader = DataLoader(dset)
    map_batch = next(iter(loader))
    assert_dicts_equal(stream_batch, map_batch)


def test_ann_spectrum_loader(mgf_small):
    """Test initialization of with a SpectrumIndex."""
    tokenizer = PeptideTokenizer()
    dset = AnnotatedSpectrumDataset(
        mgf_small,
        "seq",
        tokenizer,
        batch_size=1,
        parse_kwargs=dict(
            custom_fields=CustomField(
                "seq", lambda x: x["params"]["seq"], pa.string()
            ),
        ),
    )
    loader = DataLoader(dset, num_workers=0)
    batch = next(iter(loader))
    assert len(batch) == 8
    assert batch["mz_array"].shape == (1, 1, 13)
    assert isinstance(batch["mz_array"], torch.Tensor)
    torch.testing.assert_close(
        batch["seq"][0, ...],
        tokenizer.tokenize(["LESLIEK"], add_stop=True),
    )


def test_analyte_loader():
    """Test our peptid data loader."""
    seqs = ["LESLIE", "EDITH", "PEPTIDE"]
    charges = torch.tensor([5, 3, 1])
    tokenizer = PeptideTokenizer()
    dset = AnalyteDataset(tokenizer, seqs, charges)
    loader = DataLoader(dset, batch_size=2)

    batch = next(iter(loader))
    assert len(batch) == 2
    torch.testing.assert_close(
        batch[0],
        tokenizer.tokenize(seqs)[:2, :],
    )
    torch.testing.assert_close(batch[1], charges[:2])

    args = (torch.tensor([1, 2, 3]), torch.tensor([[1, 1], [2, 2], [3, 3]]))
    dset = AnalyteDataset(tokenizer, seqs, charges, *args)
    loader = DataLoader(dset, batch_size=2)

    batch = next(iter(loader))
    assert len(batch) == 4
    torch.testing.assert_close(batch[1], charges[:2])
    torch.testing.assert_close(batch[2], torch.tensor([1, 2]))
    torch.testing.assert_close(batch[3], args[1][:2, :])
