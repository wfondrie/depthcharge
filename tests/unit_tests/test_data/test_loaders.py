"""Test PyTorch DataLoaders"""
import torch
from depthcharge.data import (
    SpectrumIndex,
    AnnotatedSpectrumIndex,
    SpectrumDataset,
    AnnotatedSpectrumDataset,
)


def test_spectrum_index_init(mgf_small, tmp_path):
    """Test initialization of with a SpectrumIndex"""
    index = SpectrumIndex(tmp_path / "index.hdf5", mgf_small)
    dset = SpectrumDataset(index)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        collate_fn=dset.collate_fn,
        num_workers=0,
    )

    batch = next(iter(loader))
    assert len(batch) == 2
    assert batch[0].shape[0] == 1
    assert batch[0].shape[2] == 2
    assert batch[1].shape == (1, 2)

    dset = SpectrumDataset(index, n_peaks=3)
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
    """Test initialization of with a SpectrumIndex"""
    index = AnnotatedSpectrumIndex(tmp_path / "index.hdf5", mgf_small)
    dset = AnnotatedSpectrumDataset(index)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        num_workers=0,
        collate_fn=dset.collate_fn,
    )

    batch = next(iter(loader))
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
    dset = SpectrumDataset(index)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        collate_fn=dset.collate_fn,
        num_workers=0,
    )

    spec, *_ = next(iter(loader))
    assert (spec[:, :, 1] < 1).all()

    dset = SpectrumDataset(index, preprocessing_fn=[])
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        collate_fn=dset.collate_fn,
        num_workers=0,
    )

    spec, *_ = next(iter(loader))
    assert (spec[:, :, 1] == 1).all()

    def my_func(mz_array, int_array, precursor_mz, precursor_charge):
        """A simple test function"""
        int_array[:] = 2.0
        return mz_array, int_array

    dset = SpectrumDataset(index, preprocessing_fn=my_func)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        collate_fn=dset.collate_fn,
        num_workers=0,
    )

    spec, *_ = next(iter(loader))
    assert (spec[:, :, 1] == 2).all()
