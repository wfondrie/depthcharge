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

    data_module = SpectrumDataModule(
        train_index=index,
        val_index=index,
        test_index=index,
        batch_size=1,
        num_workers=0,
        n_peaks=3,
    )

    data_module.setup()
    assert data_module.train_dataloader is not None
    assert data_module.val_dataloader is not None
    assert data_module.test_dataloader is not None

    batch = next(iter(data_module.train_dataloader()))
    assert len(batch) == 2
    assert batch[0].shape[0] == 1
    assert batch[0].shape[1] == 3
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


def test_spectrum_index_reuse(mgf_small, tmp_path):
    """Reuse a previously created (annotated) spectrum index."""
    index = SpectrumIndex(tmp_path / "index.hdf5", mgf_small)
    index2 = SpectrumIndex(tmp_path / "index.hdf5")
    assert index.ms_level == index2.ms_level
    assert index.annotated == index2.annotated
    assert not index2.annotated
    assert index.n_peaks == index2.n_peaks
    assert index.n_spectra == index2.n_spectra

    index = AnnotatedSpectrumIndex(
        tmp_path / "annotated_index.hdf5", mgf_small
    )
    index2 = AnnotatedSpectrumIndex(tmp_path / "annotated_index.hdf5")
    assert index.ms_level == index2.ms_level
    assert index.annotated == index2.annotated
    assert index2.annotated
    assert index.n_peaks == index2.n_peaks
    assert index.n_spectra == index2.n_spectra


def test_preprocessing_fn(mgf_small, tmp_path):
    """Test preprocessing functions."""
    index = SpectrumIndex(tmp_path / "index.hdf5", mgf_small)
    data_module = SpectrumDataModule(
        train_index=index,
        val_index=index,
        test_index=index,
        batch_size=1,
        num_workers=0,
    )

    data_module.setup()
    spec, *_ = next(iter(data_module.train_dataloader()))
    assert (spec[:, :, 1] < 1).all()

    data_module = SpectrumDataModule(
        train_index=index,
        val_index=index,
        test_index=index,
        batch_size=1,
        preprocessing_fn=[],
        num_workers=0,
    )

    data_module.setup()
    spec, *_ = next(iter(data_module.train_dataloader()))
    assert (spec[:, :, 1] == 1).all()

    def my_func(mz_array, int_array, precursor_mz, precursor_charge):
        """A simple test function"""
        int_array[:] = 2.0
        return mz_array, int_array

    data_module = SpectrumDataModule(
        train_index=index,
        val_index=index,
        test_index=index,
        batch_size=1,
        preprocessing_fn=my_func,
        num_workers=0,
    )

    data_module.setup()
    spec, *_ = next(iter(data_module.train_dataloader()))
    assert (spec[:, :, 1] == 2).all()
