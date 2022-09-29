"""Test the datasets"""
import shutil

from depthcharge.data import (
    AnnotatedSpectrumIndex,
    SpectrumIndex,
    AnnotatedSpectrumDataset,
    SpectrumDataset,
)


def test_spectrum_id(mgf_small, tmp_path):
    """Test the mgf index."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)

    index = SpectrumIndex(tmp_path / "index.hdf5", [mgf_small, mgf_small2])
    dataset = SpectrumDataset(index)
    assert dataset.get_spectrum_id(0) == (str(mgf_small), "index=0")
    assert dataset.get_spectrum_id(3) == (str(mgf_small2), "index=1")

    index = AnnotatedSpectrumIndex(
        tmp_path / "index2.hdf5", [mgf_small, mgf_small2]
    )
    dataset = AnnotatedSpectrumDataset(index)
    assert dataset.get_spectrum_id(0) == (str(mgf_small), "index=0")
    assert dataset.get_spectrum_id(3) == (str(mgf_small2), "index=1")
