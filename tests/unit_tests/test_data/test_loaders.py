"""Test PyTorch DataLoaders."""
import torch

from depthcharge.data import (
    AnnotatedSpectrumDataset,
    SpectrumDataset,
)


def test_spectrum_index_init(mgf_small, tmp_path):
    """Test initialization of with a SpectrumIndex."""
    dset = SpectrumDataset(tmp_path / "index.hdf5", mgf_small)
    loader = dset.loader(batch_size=1, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 2
    assert batch[0].shape[0] == 1
    assert batch[0].shape[2] == 2
    assert batch[1].shape == (1, 2)

    dset = SpectrumDataset(
        tmp_path / "index.hdf5",
        mgf_small,
        preprocessing_fn=[],
        overwrite=True,
    )
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        collate_fn=dset.collate_fn,
        num_workers=0,
    )

    batch = next(iter(loader))
    assert len(batch) == 2
    assert batch[0].shape[0] == 1
    assert batch[0].shape[1] == 3
    assert batch[0].shape[2] == 2
    assert batch[1].shape == (1, 2)


def test_ann_spectrum_index_init(mgf_small, tmp_path):
    """Test initialization of with a SpectrumIndex."""
    dset = AnnotatedSpectrumDataset(tmp_path / "index.hdf5", mgf_small)
    loader = dset.loader(batch_size=1, num_workers=0)

    batch = next(iter(loader))
    assert len(batch) == 3
    assert batch[0].shape[0] == 1
    assert batch[0].shape[2] == 2
    assert batch[1].shape == (1, 2)
    assert batch[2].shape == (1,)
