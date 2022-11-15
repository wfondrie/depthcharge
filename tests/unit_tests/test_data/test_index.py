"""Test that the MS data file indices work correctly"""
import shutil
import functools

from depthcharge.data import AnnotatedSpectrumIndex, SpectrumIndex


def test_valid_charge(mgf_medium, tmp_path):
    """Test that the valid_charge argument works"""
    mkindex = functools.partial(
        SpectrumIndex,
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

    with index:
        assert index.get_spectrum_id(0) == (str(mgf_small), "index=0")
        assert index.get_spectrum_id(3) == (str(mgf_small2), "index=1")

    index = AnnotatedSpectrumIndex(
        tmp_path / "index2.hdf5", [mgf_small, mgf_small2]
    )
    assert index.ms_files == [str(mgf_small), str(mgf_small2)]
    assert index.ms_level == 2
    assert index.annotated
    assert not index.overwrite
    assert index.n_spectra == 4
    assert index.n_peaks == 92

    with index:
        assert index.get_spectrum_id(0) == (str(mgf_small), "index=0")
        assert index.get_spectrum_id(3) == (str(mgf_small2), "index=1")


def test_mzml_index(real_mzml, tmp_path):
    """Test an mzML index"""
    real_mzml2 = tmp_path / "real_mzml2.mzML"
    shutil.copy(real_mzml, real_mzml2)

    index = SpectrumIndex(tmp_path / "index.hdf5", [real_mzml, real_mzml2])
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
    index = SpectrumIndex(tmp_path / "index2.hdf5", [real_mzml, real_mzml2], 3)
    assert index.ms_files == [str(real_mzml), str(real_mzml2)]
    assert index.ms_level == 3
    assert index.n_spectra == 6
    assert index.n_peaks == 194

    with index:
        assert index.get_spectrum_id(0) == (str(real_mzml), "scan=502")
        assert index.get_spectrum_id(5) == (str(real_mzml2), "scan=508")

    # MS1
    index = SpectrumIndex(tmp_path / "index3.hdf5", [real_mzml, real_mzml2], 1)
    assert index.ms_files == [str(real_mzml), str(real_mzml2)]
    assert index.ms_level == 1
    assert index.n_spectra == 8
    assert index.n_peaks == 4316

    with index:
        assert index.get_spectrum_id(0) == (str(real_mzml), "scan=500")
        assert index.get_spectrum_id(5) == (str(real_mzml2), "scan=503")


def test_mzxml_index(real_mzxml, tmp_path):
    """Test an mzXML index"""
    real_mzxml2 = tmp_path / "real_mzxml2.mzXML"
    shutil.copy(real_mzxml, real_mzxml2)

    index = SpectrumIndex(tmp_path / "index.hdf5", [real_mzxml, real_mzxml2])
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
    index = SpectrumIndex(
        tmp_path / "index2.hdf5", [real_mzxml, real_mzxml2], 3
    )
    assert index.ms_files == [str(real_mzxml), str(real_mzxml2)]
    assert index.ms_level == 3
    assert index.n_spectra == 6
    assert index.n_peaks == 194

    with index:
        assert index.get_spectrum_id(0) == (str(real_mzxml), "scan=502")
        assert index.get_spectrum_id(5) == (str(real_mzxml2), "scan=508")

    # MS1
    index = SpectrumIndex(
        tmp_path / "index3.hdf5", [real_mzxml, real_mzxml2], 1
    )
    assert index.ms_files == [str(real_mzxml), str(real_mzxml2)]
    assert index.ms_level == 1
    assert index.n_spectra == 8
    assert index.n_peaks == 4316

    with index:
        assert index.get_spectrum_id(0) == (str(real_mzxml), "scan=500")
        assert index.get_spectrum_id(5) == (str(real_mzxml2), "scan=503")
