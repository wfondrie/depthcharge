"""Test the arrow functionality."""
import polars as pl
import pytest

from depthcharge.data.arrow import (
    spectra_to_df,
    spectra_to_parquet,
    spectra_to_stream,
)
from depthcharge.data.preprocessing import scale_to_unit_norm

METADATA_DF1 = pl.DataFrame({"scan_id": [501, 507, 10], "blah": [True] * 3})
METADATA_DF2 = pl.DataFrame(
    {
        "scan_id": [501, 507, 10],
        "blah": [True] * 3,
        "peak_file": ["TMT10-Trail-8.mzML"] * 3,
    },
)
PARAM_NAMES = [
    "ms_level",
    "preprocessing_fn",
    "valid_charge",
    "custom_fields",
    "metadata_df",
    "shape",
]
PARAM_VALS = [
    (2, None, None, None, None, (4, 7)),
    (1, None, None, None, None, (4, 7)),
    (3, None, None, None, None, (3, 7)),
    (2, None, [3], None, None, (3, 7)),
    (None, None, None, None, None, (11, 7)),
    (2, scale_to_unit_norm, None, {"index": ["index"]}, None, (4, 8)),
    (2, None, None, None, METADATA_DF1, (4, 8)),
    (2, None, None, None, METADATA_DF2, (4, 8)),
]


@pytest.mark.parametrize(PARAM_NAMES, PARAM_VALS)
def test_to_df(
    real_mzml,
    ms_level,
    preprocessing_fn,
    valid_charge,
    custom_fields,
    metadata_df,
    shape,
):
    """Test parsing to DataFrame."""
    parsed = spectra_to_df(
        real_mzml,
        ms_level=ms_level,
        preprocessing_fn=preprocessing_fn,
        valid_charge=valid_charge,
        custom_fields=custom_fields,
        metadata_df=metadata_df,
    )

    assert parsed.shape == shape


@pytest.mark.parametrize(PARAM_NAMES, PARAM_VALS)
def test_to_parquet(
    tmp_path,
    real_mzml,
    ms_level,
    preprocessing_fn,
    valid_charge,
    custom_fields,
    shape,
):
    """Test parsing to DataFrame."""
    out = spectra_to_parquet(
        real_mzml,
        parquet_file=tmp_path / "test",
        ms_level=ms_level,
        preprocessing_fn=preprocessing_fn,
        valid_charge=valid_charge,
        custom_fields=custom_fields,
    )

    parsed = pl.read_parquet(out)
    assert parsed.shape == shape


@pytest.mark.parametrize(PARAM_NAMES, PARAM_VALS)
def test_to_stream(
    real_mzml, ms_level, preprocessing_fn, valid_charge, custom_fields, shape
):
    """Test parsing to DataFrame."""
    out = spectra_to_stream(
        real_mzml,
        batch_size=1,
        ms_level=ms_level,
        preprocessing_fn=preprocessing_fn,
        valid_charge=valid_charge,
        custom_fields=custom_fields,
    )

    out = list(out)
    assert len(out) == shape[0]

    parsed = pl.from_arrow(out)
    assert parsed.shape == shape
