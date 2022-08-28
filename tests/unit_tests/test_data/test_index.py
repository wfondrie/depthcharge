"""Test that the MS data file indices work correctly"""
import shutil

from depthcharge.data import AnnotatedSpectrumIndex, SpectrumIndex


def test_mgf_index(mgf_small, tmp_path):
    """Test the mgf index."""
    mgf_small2 = tmp_path / "mgf_small2.mgf"
    shutil.copy(mgf_small, mgf_small2)

    index = SpectrumIndex(tmp_path / "index.hdf5", [mgf_small, mgf_small2])
    assert index.ms_files == [str(mgf_small), str(mgf_small2)]
    assert index.ms_level == 2
    assert not index.annotated
    assert not index.overwrite
    assert index.n_spectra == 4
    assert index.n_peaks == 92

    with index as myidx:
        assert myidx.get_spectrum_id(0) == (str(mgf_small), "index=0")
