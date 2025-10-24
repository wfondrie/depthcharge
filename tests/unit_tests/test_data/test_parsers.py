"""Test that parsers work."""

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from depthcharge.data import CustomField
from depthcharge.data.parsers import (
    MgfParser,
    MzmlParser,
    MzxmlParser,
    ParserFactory,
    TdfParser,
)
from depthcharge.data.preprocessing import scale_to_unit_norm

SMALL_MGF_MZS = [
    [
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
    ],
    [
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
    ],
]

MGF_FIELD = CustomField("t", lambda x: x["params"]["title"], pa.string())
MZML_FIELD = CustomField("index", lambda x: x["index"], pa.int64())
MZXML_FIELD = CustomField("CE", lambda x: x["collisionEnergy"], pa.float64())


def test_mgf_and_base(mgf_small):
    """MGF file with a missing charge."""
    parsed = pl.from_arrow(
        MgfParser(mgf_small, preprocessing_fn=[]).iter_batches(None)
    )
    expected = pl.DataFrame(
        {
            "peak_file": [mgf_small.name] * 2,
            "scan_id": ["0", "1"],
            "ms_level": [2, 2],
            "precursor_mz": [416.24474357, 257.464565],
            "precursor_charge": [2, 3],
            "mz_array": SMALL_MGF_MZS,
            "intensity_array": [
                [1.0] * len(SMALL_MGF_MZS[0]),
                [1.0] * len(SMALL_MGF_MZS[1]),
            ],
        }
    ).with_columns(
        [
            pl.col("intensity_array").cast(pl.List(pl.Float64)),
            pl.col("ms_level").cast(pl.UInt8),
            pl.col("precursor_charge").cast(pl.Int16),
        ]
    )

    assert parsed.shape == (2, 7)
    assert_frame_equal(parsed, expected)

    parsed = pl.from_arrow(
        MgfParser(mgf_small, valid_charge=[2]).iter_batches(2),
    )
    assert parsed.shape == (1, 7)
    assert isinstance(ParserFactory.get_parser(mgf_small), MgfParser)


@pytest.mark.parametrize(
    ["ms_level", "preprocessing_fn", "valid_charge", "custom_fields", "shape"],
    [
        (2, None, None, None, (4, 7)),
        (1, None, None, None, (4, 7)),
        (3, None, None, None, (3, 7)),
        (2, None, [3], None, (3, 7)),
        (None, None, None, None, (11, 7)),
        (2, scale_to_unit_norm, None, MZML_FIELD, (4, 8)),
    ],
)
def test_mzml(
    real_mzml, ms_level, preprocessing_fn, valid_charge, custom_fields, shape
):
    """A simple mzML test."""
    parsed = pl.from_arrow(
        MzmlParser(
            real_mzml,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            custom_fields=custom_fields,
        ).iter_batches(None)
    )
    assert parsed.shape == shape


@pytest.mark.parametrize(
    ["ms_level", "preprocessing_fn", "valid_charge", "custom_fields", "shape"],
    [
        (2, None, None, None, (4, 7)),
        (1, None, None, None, (4, 7)),
        (3, None, None, None, (3, 7)),
        (2, None, [3], None, (3, 7)),
        (None, None, None, None, (11, 7)),
        (2, scale_to_unit_norm, None, MZXML_FIELD, (4, 8)),
    ],
)
def test_mzxml(
    real_mzxml, ms_level, preprocessing_fn, valid_charge, custom_fields, shape
):
    """A simple mzML test."""
    parsed = pl.from_arrow(
        MzxmlParser(
            real_mzxml,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            custom_fields=custom_fields,
        ).iter_batches(None)
    )
    assert parsed.shape == shape


@pytest.mark.parametrize(
    ["ms_level", "preprocessing_fn", "valid_charge", "custom_fields", "shape"],
    [
        (2, None, None, None, (7, 7)),
        (1, None, None, None, (7, 7)),
        (3, None, None, None, (7, 7)),
        (2, None, [3], None, (3, 7)),
        (None, None, None, None, (7, 7)),
        (2, scale_to_unit_norm, None, MGF_FIELD, (7, 8)),
    ],
)
def test_mgf(
    real_mgf, ms_level, preprocessing_fn, valid_charge, custom_fields, shape
):
    """A simple mzML test."""
    parsed = pl.from_arrow(
        MgfParser(
            real_mgf,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            custom_fields=custom_fields,
        ).iter_batches(None)
    )
    assert parsed.shape == shape


@pytest.mark.parametrize(
    ["ms_level", "preprocessing_fn", "valid_charge", "custom_fields", "shape"],
    [
        (2, None, None, None, (3, 7)),
        (2, None, [3], None, (1, 7)),
        (None, None, None, None, (3, 7)),
        (None, scale_to_unit_norm, None, None, (3, 7)),
    ],
)
def test_tdf(
    real_tdf, ms_level, preprocessing_fn, valid_charge, custom_fields, shape
):
    """A simple TDF test."""
    parsed = pl.from_arrow(
        TdfParser(
            real_tdf,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            custom_fields=custom_fields,
        ).iter_batches(None)
    )
    assert parsed.shape == shape


def test_custom_fields(mgf_small):
    """Test that custom fields are working."""
    parsed = pl.from_arrow(
        MgfParser(
            mgf_small,
            custom_fields=CustomField(
                "seq", lambda x: x["params"]["seq"], pa.string()
            ),
        ).iter_batches(None)
    )

    expected = pl.Series("seq", ["LESLIEK", "EDITHR"])
    assert_series_equal(parsed["seq"], expected)

    with pytest.raises(KeyError):
        pl.from_arrow(
            MgfParser(
                mgf_small,
                custom_fields=CustomField(
                    "seq", lambda x: x["params"]["bar"], pa.string()
                ),
            ).iter_batches(None)
        )


def test_invalid_file(tmp_path):
    """Test an invalid file raises an error."""
    tmp_path.touch("blah.txt")

    with pytest.raises(OSError):
        ParserFactory().get_parser(tmp_path / "blah.txt")
