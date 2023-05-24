"""Test that parsers work."""

import numpy as np

from depthcharge.data.parsers import MgfParser, MzmlParser, MzxmlParser

SMALL_MGF_MZS = [
    114.09134044,
    147.11280416,
    243.13393353,
    276.15539725,
    330.16596194,
    389.23946123,
    443.25002591,
    502.32352521,
    556.33408989,
    589.35555361,
    685.37668298,
    718.3981467,
    813.47164599,
    831.48221068,
    65.52857301,
    88.06311432,
    123.04204452,
    130.04986955,
    156.59257025,
    175.11895217,
    179.58407651,
    207.11640948,
    230.10791575,
    245.07681258,
    263.65844147,
    298.63737167,
    312.17786403,
    321.17191298,
    358.16087656,
    376.68792719,
    385.69320953,
    413.2255425,
    459.20855502,
    526.30960648,
    596.26746688,
    641.3365495,
    752.36857791,
    770.37914259,
]


def test_mgf_and_base(mgf_small):
    """MGF file with a missing charge."""
    parser = MgfParser(mgf_small).read()
    np.testing.assert_allclose(
        parser.precursor_mz,
        np.array([416.24474357, 257.464565]),
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

    np.testing.assert_allclose(
        parser.intensity_arrays, np.ones(len(SMALL_MGF_MZS), dtype=int)
    )

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
    """A simple mzML test."""
    parser = MzmlParser(real_mzml, 2).read()
    prev_len = len(parser.mz_arrays)
    assert len(parser.intensity_arrays) == prev_len
    assert len(parser.precursor_charge) == 4

    parser = MzmlParser(real_mzml, 2, valid_charge=[3]).read()
    assert len(parser.precursor_charge) == 3
    assert len(parser.intensity_arrays) < prev_len
    assert len(parser.mz_arrays) < prev_len


def test_mzxml(real_mzxml):
    """A simple mzML test."""
    parser = MzxmlParser(real_mzxml, 2).read()
    prev_len = len(parser.mz_arrays)
    assert len(parser.intensity_arrays) == prev_len
    assert len(parser.precursor_charge) == 4

    parser = MzxmlParser(real_mzxml, 2, valid_charge=[3]).read()
    assert len(parser.precursor_charge) == 3
    assert len(parser.intensity_arrays) < prev_len
    assert len(parser.mz_arrays) < prev_len
