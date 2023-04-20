"""Test that parsers work."""

import numpy as np
from depthcharge.data.parsers import MzmlParser, MgfParser, MzxmlParser

SMALL_MGF_MZS = [
    114.09134044,
    831.48221068,
    243.13393353,
    718.3981467,
    330.16596194,
    589.35555361,
    443.25002591,
    502.32352521,
    556.33408989,
    389.23946123,
    685.37668298,
    276.15539725,
    813.47164599,
    147.11280416,
    98.06004032,
    928.46220351,
    49.53365839,
    464.73473999,
    227.1026334,
    831.40943966,
    114.05495494,
    416.20835806,
    324.15539725,
    702.36684657,
    162.58133686,
    351.68706152,
    425.20307572,
    605.31408272,
    213.10517609,
    303.16067959,
    538.2871397,
    504.26640425,
    269.64720808,
    252.63684036,
    653.31408272,
    391.18234028,
    327.16067959,
    196.09480837,
    782.35667581,
    276.15539725,
    391.68197614,
    138.58133686,
    910.45163882,
    147.11280416,
    455.72945765,
    74.06004032,
]


def test_mgf_and_base(mgf_small):
    """MGF file with a missing charge"""
    parser = MgfParser(mgf_small).read()
    np.testing.assert_allclose(
        parser.precursor_mz,
        np.array([416.24474357, 310.15891881]),
    )
    np.testing.assert_equal(
        parser.precursor_charge,
        np.array([2, 3]),
    )
    np.testing.assert_equal(
        parser.offset,
        np.array([0, 14]),
    )
    np.testing.assert_allclose(
        parser.mz_arrays,
        np.array(SMALL_MGF_MZS),
    )

    np.testing.assert_allclose(parser.intensity_arrays, np.ones(46, dtype=int))

    parser = MgfParser(mgf_small, valid_charge=[2]).read()
    assert parser.precursor_charge.shape == (1,)

    parser = MgfParser(mgf_small, ms_level=1).read()
    np.testing.assert_equal(
        parser.precursor_charge,
        np.array([0, 0]),
    )
    np.testing.assert_equal(
        parser.precursor_mz,
        np.array([np.nan, np.nan]),
    )


def test_mzml(real_mzml):
    """A simple mzML test"""
    parser = MzmlParser(real_mzml).read()
    prev_len = len(parser.mz_arrays)
    assert len(parser.intensity_arrays) == prev_len
    assert len(parser.precursor_charge) == 4

    parser = MzmlParser(real_mzml, valid_charge=[3]).read()
    assert len(parser.precursor_charge) == 3
    assert len(parser.intensity_arrays) < prev_len
    assert len(parser.mz_arrays) < prev_len


def test_mzxml(real_mzxml):
    """A simple mzML test"""
    parser = MzxmlParser(real_mzxml).read()
    prev_len = len(parser.mz_arrays)
    assert len(parser.intensity_arrays) == prev_len
    assert len(parser.precursor_charge) == 4

    parser = MzxmlParser(real_mzxml, valid_charge=[3]).read()
    assert len(parser.precursor_charge) == 3
    assert len(parser.intensity_arrays) < prev_len
    assert len(parser.mz_arrays) < prev_len
