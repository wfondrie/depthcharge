"""Test PyTorch DataLoaders."""
import numpy as np
import torch

from depthcharge.data import (
    AnnotatedSpectrumDataset,
    PeptideDataset,
    SpectrumDataset,
)


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
    dset = AnnotatedSpectrumDataset(mgf_small)
    loader = dset.loader(batch_size=1, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 3
    assert batch[0].shape == (1, 13, 2)
    assert batch[1].shape == (1, 2)
    assert batch[2].shape == (1,)


def test_peptide_loaer():
    """Test our peptid data loader."""
    seqs = ["LESLIE", "TOBIN", "EDITH"]
    charges = [5, 3, 1]
    dset = PeptideDataset(seqs, charges)
    loader = dset.loader(batch_size=2, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 2
    np.testing.assert_equal(batch[0], np.array(seqs[:2]))
    torch.testing.assert_close(batch[1], torch.tensor(charges[:2], dtype=int))
