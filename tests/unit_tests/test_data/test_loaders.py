"""Test PyTorch DataLoaders and PyTorch-Lightning LightningDataModules"""
from depthcharge.data import (
    SpectrumDataModule,
    SpectrumIndex,
    AnnotatedSpectrumIndex,
)


def test_spectrum_index_init(mgf_small, tmp_path):
    """Test initialization of with a SpectrumIndex"""
    index = SpectrumIndex(tmp_path / "index.hdf5", mgf_small)
    data_module = SpectrumDataModule(
        train_index=index,
        val_index=index,
        test_index=index,
        batch_size=1,
        num_workers=0,
    )

    data_module.setup()
    assert data_module.train_dataloader is not None
    assert data_module.val_dataloader is not None
    assert data_module.test_dataloader is not None

    batch = next(iter(data_module.train_dataloader()))
    assert len(batch) == 2
    assert batch[0].shape[0] == 1
    assert batch[0].shape[2] == 2
    assert batch[1].shape == (1, 2)


def test_ann_spectrum_index_init(mgf_small, tmp_path):
    """Test initialization of with a SpectrumIndex"""
    index = AnnotatedSpectrumIndex(tmp_path / "index.hdf5", mgf_small)
    data_module = SpectrumDataModule(
        train_index=index,
        val_index=index,
        test_index=index,
        batch_size=1,
        num_workers=0,
    )

    data_module.setup()
    assert data_module.train_dataloader is not None
    assert data_module.val_dataloader is not None
    assert data_module.test_dataloader is not None

    batch = next(iter(data_module.train_dataloader()))
    assert len(batch) == 3
    assert batch[0].shape[0] == 1
    assert batch[0].shape[2] == 2
    assert batch[1].shape == (1, 2)
    assert batch[2].shape == (1,)
