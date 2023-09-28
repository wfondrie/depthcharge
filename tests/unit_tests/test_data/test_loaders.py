"""Test PyTorch DataLoaders."""
import pytest
import torch

from depthcharge.data import (
    AnnotatedSpectrumDataset,
    PeptideDataset,
    SpectrumDataset,
    StreamingSpectrumDataset,
)
from depthcharge.testing import assert_dicts_equal
from depthcharge.tokenizers import PeptideTokenizer


def test_spectrum_loader(mgf_small, tmp_path):
    """Test a normal spectrum dataset."""
    dset = SpectrumDataset(mgf_small, tmp_path / "test")
    loader = dset.loader(batch_size=1, num_workers=0)
    batch = next(iter(loader))
    assert len(batch) == 7
    assert batch["mz_array"].shape == (1, 13)
    assert isinstance(batch["mz_array"], torch.Tensor)

    with pytest.warns(UserWarning):
        dset.loader(collate_fn=torch.utils.data.default_collate)


def test_streaming_spectrum_loader(mgf_small, tmp_path):
    """Test streaming spectra."""
    streamer = StreamingSpectrumDataset(mgf_small, 2)
    loader = streamer.loader(batch_size=2, num_workers=0)
    stream_batch = next(iter(loader))
    with pytest.warns(UserWarning):
        streamer.loader(collate_fn=torch.utils.data.default_collate)

    with pytest.warns(UserWarning):
        streamer.loader(num_workers=2)

    dset = SpectrumDataset(mgf_small, tmp_path / "test")
    loader = dset.loader(batch_size=2, num_workers=0)
    map_batch = next(iter(loader))
    assert_dicts_equal(stream_batch, map_batch)


def test_ann_spectrum_loader(mgf_small):
    """Test initialization of with a SpectrumIndex."""
    tokenizer = PeptideTokenizer()
    dset = AnnotatedSpectrumDataset(
        mgf_small, "seq", tokenizer, custom_fields={"seq": ["params", "seq"]}
    )
    loader = dset.loader(batch_size=1, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 8
    assert batch["mz_array"].shape == (1, 13)
    assert isinstance(batch["mz_array"], torch.Tensor)
    torch.testing.assert_close(batch["seq"], tokenizer.tokenize(["LESLIEK"]))

    with pytest.warns(UserWarning):
        dset.loader(collate_fn=torch.utils.data.default_collate)


def test_peptide_loader():
    """Test our peptid data loader."""
    seqs = ["LESLIE", "EDITH", "PEPTIDE"]
    charges = torch.tensor([5, 3, 1])
    tokenizer = PeptideTokenizer()
    dset = PeptideDataset(tokenizer, seqs, charges)
    loader = dset.loader(batch_size=2, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 2
    torch.testing.assert_close(
        batch[0],
        tokenizer.tokenize(seqs)[:2, :],
    )
    torch.testing.assert_close(batch[1], charges[:2])

    args = (torch.tensor([1, 2, 3]), torch.tensor([[1, 1], [2, 2], [3, 3]]))
    dset = PeptideDataset(tokenizer, seqs, charges, *args)
    loader = dset.loader(batch_size=2, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 4
    torch.testing.assert_close(batch[1], charges[:2])
    torch.testing.assert_close(batch[2], torch.tensor([1, 2]))
    torch.testing.assert_close(batch[3], args[1][:2, :])
