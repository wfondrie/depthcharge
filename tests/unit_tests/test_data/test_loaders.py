"""Test PyTorch DataLoaders."""
import torch

from depthcharge.data import (
    AnnotatedSpectrumDataset,
    PeptideDataset,
    SpectrumDataset,
)
from depthcharge.tokenizers import PeptideTokenizer


def test_spectrum_loader(mgf_small):
    """Test initialization of with a SpectrumIndex."""
    dset = SpectrumDataset(mgf_small, 2)
    loader = dset.loader(batch_size=1, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 2
    assert batch[0].shape == (1, 13, 2)
    assert batch[1].shape == (1, 2)

    dset = SpectrumDataset(mgf_small, 2, [])
    loader = dset.loader(batch_size=1, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 2
    assert batch[0].shape == (1, 14, 2)
    assert batch[1].shape == (1, 2)


def test_ann_spectrum_loader(mgf_small):
    """Test initialization of with a SpectrumIndex."""
    tokenizer = PeptideTokenizer()
    dset = AnnotatedSpectrumDataset(tokenizer, mgf_small)
    loader = dset.loader(batch_size=1, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 3
    assert batch[0].shape == (1, 13, 2)
    assert batch[1].shape == (1, 2)
    assert batch[2].shape == (1, 7)


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
    torch.testing.assert_close(batch[1], torch.tensor(charges[:2], dtype=int))

    args = (torch.tensor([1, 2, 3]), torch.tensor([[1, 1], [2, 2], [3, 3]]))
    dset = PeptideDataset(tokenizer, seqs, charges, *args)
    loader = dset.loader(batch_size=2, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 4
    torch.testing.assert_close(batch[1], torch.tensor(charges[:2], dtype=int))
    torch.testing.assert_close(batch[2], torch.tensor([1, 2]))
    torch.testing.assert_close(batch[3], args[1][:2, :])
