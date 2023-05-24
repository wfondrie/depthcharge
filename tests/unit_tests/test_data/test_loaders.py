"""Test PyTorch DataLoaders."""

from depthcharge.data import (
    AnnotatedSpectrumDataset,
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
