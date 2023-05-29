"""Test the datasets."""
import functools
import shutil

import pytest
import torch

from depthcharge.data import (
    AnnotatedSpectrumDataset,
    PeptideDataset,
    SpectrumDataset,
)
from depthcharge.primitives import MassSpectrum
from depthcharge.tokenizers import PeptideTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    """Use a tokenizer for every test."""
    return PeptideTokenizer()


def test_spectrum_id(mgf_small, tmp_path):
    """Test the mgf index."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)

    dataset = SpectrumDataset([mgf_small, mgf_small2])
    with dataset:
        assert dataset.get_spectrum_id(0) == (str(mgf_small), "index=0")
        assert dataset.get_spectrum_id(3) == (str(mgf_small2), "index=1")

    dataset = AnnotatedSpectrumDataset(tokenizer, [mgf_small, mgf_small2])
    with dataset:
        assert dataset.get_spectrum_id(0) == (str(mgf_small), "index=0")
        assert dataset.get_spectrum_id(3) == (str(mgf_small2), "index=1")


def test_indexing(mgf_small, tmp_path):
    """Test retrieving spectra."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)

    dataset = SpectrumDataset([mgf_small, mgf_small2])
    spec = dataset[0]
    assert isinstance(spec, MassSpectrum)
    assert spec.label is None

    dataset = AnnotatedSpectrumDataset(tokenizer, [mgf_small, mgf_small2])
    spec = dataset[0]
    assert isinstance(spec, MassSpectrum)
    assert spec.label == "LESLIEK"
    assert spec.to_tensor().shape[1] == 2
    assert dataset[3].label == "EDITHR"


def test_valid_charge(mgf_medium, tmp_path):
    """Test that the valid_charge argument works."""
    mkindex = functools.partial(
        SpectrumDataset,
        index_path=tmp_path / "index.hdf5",
        ms_data_files=mgf_medium,
        overwrite=True,
    )

    index = mkindex()
    assert index.n_spectra == 100

    index = mkindex(valid_charge=[0, 2, 3, 4])
    assert index.n_spectra == 100

    index = mkindex(valid_charge=[2, 3, 4])
    assert index.n_spectra == 99

    index = mkindex(valid_charge=[2, 3])
    assert index.n_spectra == 98


def test_mgf(mgf_small, tmp_path):
    """Test the mgf index."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)

    index = SpectrumDataset([mgf_small, mgf_small2])
    assert index.ms_files == [str(mgf_small), str(mgf_small2)]
    assert index.ms_level == 2
    assert not index.annotated
    assert not index.overwrite
    assert index.n_spectra == 4
    assert index.n_peaks == 66

    with index:
        assert index.get_spectrum_id(0) == (str(mgf_small), "index=0")
        assert index.get_spectrum_id(3) == (str(mgf_small2), "index=1")

    index = AnnotatedSpectrumDataset(tokenizer, [mgf_small, mgf_small2])
    assert index.ms_files == [str(mgf_small), str(mgf_small2)]
    assert index.ms_level == 2
    assert index.annotated
    assert not index.overwrite
    assert index.n_spectra == 4
    assert index.n_peaks == 66

    with index:
        assert index.get_spectrum_id(0) == (str(mgf_small), "index=0")
        assert index.get_spectrum_id(3) == (str(mgf_small2), "index=1")


def test_mzml_index(real_mzml, tmp_path):
    """Test an mzML index."""
    real_mzml2 = tmp_path / "real_mzml2.mzML"
    shutil.copy(real_mzml, real_mzml2)

    index = SpectrumDataset([real_mzml, real_mzml2], 2, [])
    assert index.ms_files == [str(real_mzml), str(real_mzml2)]
    assert index.ms_level == 2
    assert not index.annotated
    assert not index.overwrite
    assert index.n_spectra == 8
    assert index.n_peaks == 726

    with index:
        assert index.get_spectrum_id(0) == (str(real_mzml), "scan=501")
        assert index.get_spectrum_id(7) == (str(real_mzml2), "scan=510")

    # MS3
    index = SpectrumDataset([real_mzml, real_mzml2], 3, [])
    assert index.ms_files == [str(real_mzml), str(real_mzml2)]
    assert index.ms_level == 3
    assert index.n_spectra == 6
    assert index.n_peaks == 194

    with index:
        assert index.get_spectrum_id(0) == (str(real_mzml), "scan=502")
        assert index.get_spectrum_id(5) == (str(real_mzml2), "scan=508")

    # MS1
    index = SpectrumDataset([real_mzml, real_mzml2], 1, [])
    assert index.ms_files == [str(real_mzml), str(real_mzml2)]
    assert index.ms_level == 1
    assert index.n_spectra == 8
    assert index.n_peaks == 4316

    with index:
        assert index.get_spectrum_id(0) == (str(real_mzml), "scan=500")
        assert index.get_spectrum_id(5) == (str(real_mzml2), "scan=503")


def test_mzxml_index(real_mzxml, tmp_path):
    """Test an mzXML index."""
    real_mzxml2 = tmp_path / "real_mzxml2.mzXML"
    shutil.copy(real_mzxml, real_mzxml2)

    index = SpectrumDataset([real_mzxml, real_mzxml2], 2, [])
    assert index.ms_files == [str(real_mzxml), str(real_mzxml2)]
    assert index.ms_level == 2
    assert not index.annotated
    assert not index.overwrite
    assert index.n_spectra == 8
    assert index.n_peaks == 726

    with index:
        assert index.get_spectrum_id(0) == (str(real_mzxml), "scan=501")
        assert index.get_spectrum_id(7) == (str(real_mzxml2), "scan=510")

    # MS3
    index = SpectrumDataset([real_mzxml, real_mzxml2], 3, [])
    assert index.ms_files == [str(real_mzxml), str(real_mzxml2)]
    assert index.ms_level == 3
    assert index.n_spectra == 6
    assert index.n_peaks == 194

    with index:
        assert index.get_spectrum_id(0) == (str(real_mzxml), "scan=502")
        assert index.get_spectrum_id(5) == (str(real_mzxml2), "scan=508")

    # MS1
    index = SpectrumDataset([real_mzxml, real_mzxml2], 1, [])
    assert index.ms_files == [str(real_mzxml), str(real_mzxml2)]
    assert index.ms_level == 1
    assert index.n_spectra == 8
    assert index.n_peaks == 4316

    with index:
        assert index.get_spectrum_id(0) == (str(real_mzxml), "scan=500")
        assert index.get_spectrum_id(5) == (str(real_mzxml2), "scan=503")


def test_spectrum_index_reuse(mgf_small, tmp_path):
    """Reuse a previously created (annotated) spectrum index."""
    plain_index = tmp_path / "plain.hdf5"
    ann_index = tmp_path / "ann.hdf5"

    index = SpectrumDataset(mgf_small, index_path=plain_index)
    index2 = SpectrumDataset(index_path=plain_index)
    assert index.ms_level == index2.ms_level
    assert index.annotated == index2.annotated
    assert not index2.annotated
    assert index.n_peaks == index2.n_peaks
    assert index.n_spectra == index2.n_spectra

    index3 = AnnotatedSpectrumDataset(
        tokenizer, mgf_small, index_path=ann_index
    )
    index4 = AnnotatedSpectrumDataset(tokenizer, index_path=ann_index)
    assert index3.ms_level == index4.ms_level
    assert index3.annotated == index4.annotated
    assert index4.annotated
    assert index3.n_peaks == index4.n_peaks
    assert index3.n_spectra == index4.n_spectra

    # An annotated spectrum dataset may be loaded as a spectrum dataset,
    # but not vice versa:
    SpectrumDataset(index_path=ann_index)
    with pytest.raises(ValueError):
        AnnotatedSpectrumDataset(tokenizer, index_path=plain_index)

    # Verify we invalidate correctly:
    with pytest.raises(ValueError):
        SpectrumDataset(index_path=plain_index, ms_level=1)

    with pytest.raises(ValueError):
        SpectrumDataset(index_path=plain_index, preprocessing_fn=[])

    with pytest.raises(ValueError):
        AnnotatedSpectrumDataset(tokenizer, index_path=ann_index, ms_level=3)

    with pytest.raises(ValueError):
        AnnotatedSpectrumDataset(
            tokenizer, index_path=plain_index, preprocessing_fn=[]
        )


def test_spectrum_indexing_bug(tmp_path, mgf_small):
    """Test that we've fixed reindexing upon reload."""
    dset1 = AnnotatedSpectrumDataset(
        PeptideTokenizer(), mgf_small, 2, index_path=tmp_path / "test.hdf5"
    )

    dset2 = AnnotatedSpectrumDataset(
        PeptideTokenizer(), mgf_small, 2, index_path=tmp_path / "test.hdf5"
    )

    assert dset1._locs == dset2._locs


def test_preprocessing_fn(mgf_small):
    """Test preprocessing functions."""
    dset = SpectrumDataset(mgf_small)
    loader = dset.loader(batch_size=1, num_workers=0)

    spec, *_ = next(iter(loader))
    assert (spec[:, :, 1] < 1).all()

    dset = SpectrumDataset(mgf_small, preprocessing_fn=[])
    loader = dset.loader(batch_size=1, num_workers=0)

    spec, *_ = next(iter(loader))
    assert (spec[:, :, 1] == 1).all()

    def my_func(spec):
        """A simple test function."""
        spec.intensity[:] = 2.0
        return spec

    dset = SpectrumDataset(mgf_small, preprocessing_fn=my_func)
    loader = dset.loader(batch_size=1, num_workers=0)
    spec, *_ = next(iter(loader))
    assert (spec[:, :, 1] == 2).all()


def test_peptide_dataset():
    """Test the peptide dataset."""
    tokenizer = PeptideTokenizer()
    seqs = ["LESLIEK", "EDITHR"]
    charges = torch.tensor([2, 3])
    dset = PeptideDataset(tokenizer, seqs, charges)
    torch.testing.assert_close(dset[0][0], tokenizer.tokenize("LESLIEK"))
    torch.testing.assert_close(dset[1][0][:6], tokenizer.tokenize("EDITHR"))
    assert dset[0][1].item() == 2
    assert dset[1][1].item() == 3
    assert len(dset) == 2

    seqs = ["LESLIEK", "EDITHR"]
    charges = torch.tensor([2, 3])
    target = torch.tensor([1.1, 2.2])
    other = torch.tensor([[1, 1], [2, 2]])
    dset = PeptideDataset(tokenizer, seqs, charges, target, other)
    torch.testing.assert_close(dset[0][0], tokenizer.tokenize("LESLIEK"))
    torch.testing.assert_close(dset[1][0][:6], tokenizer.tokenize("EDITHR"))
    assert dset[0][1].item() == 2
    assert dset[1][1].item() == 3
    torch.testing.assert_close(dset[0][2], torch.tensor(1.1))
    torch.testing.assert_close(dset[1][3], other[1, :])
    assert len(dset) == 2

    torch.testing.assert_close(dset.tokens, tokenizer.tokenize(seqs))
    torch.testing.assert_close(dset.charges, charges)
