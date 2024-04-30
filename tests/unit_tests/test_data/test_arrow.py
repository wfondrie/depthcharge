"""Test the arrow functionality."""

import os

import polars as pl
import pyarrow as pa
import pytest

from depthcharge.data.arrow import (
    spectra_to_df,
    spectra_to_parquet,
    spectra_to_stream,
)
from depthcharge.data.fields import CustomField
from depthcharge.data.preprocessing import scale_to_unit_norm

SCANS = [
    f"controllerType=0 controllerNumber=1 scan={x}" for x in [501, 507, 10]
]
METADATA_DF1 = pl.DataFrame({"scan_id": SCANS, "blah": [True] * 3})
METADATA_DF2 = pl.DataFrame(
    {
        "scan_id": SCANS,
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
    "progress",
    "shape",
]

custom_field = CustomField("index", lambda x: x["index"], pa.int64())
PARAM_VALS = [
    (2, None, None, None, None, True, (4, 7)),
    (1, None, None, None, None, True, (4, 7)),
    (3, None, None, None, None, True, (3, 7)),
    (2, None, [3], None, None, True, (3, 7)),
    (None, None, None, None, None, True, (11, 7)),
    (2, scale_to_unit_norm, None, custom_field, None, True, (4, 8)),
    (2, None, None, None, METADATA_DF1, True, (4, 8)),
    (2, None, None, None, METADATA_DF2, False, (4, 8)),
]


@pytest.mark.parametrize(PARAM_NAMES, PARAM_VALS)
def test_to_df(
    real_mzml,
    ms_level,
    preprocessing_fn,
    valid_charge,
    custom_fields,
    metadata_df,
    progress,
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
        progress=progress,
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
    metadata_df,
    progress,
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
        metadata_df=metadata_df,
        progress=progress,
    )

    parsed = pl.read_parquet(out)
    assert parsed.shape == shape


def test_to_parquet_with_default_file(real_mzml, tmp_path):
    """Test that a default file can be created."""
    os.chdir(tmp_path)

    out = spectra_to_parquet(
        real_mzml,
        parquet_file=None,
        progress=False,
    )

    assert out.exists()


@pytest.mark.parametrize(PARAM_NAMES, PARAM_VALS)
def test_to_stream(
    real_mzml,
    ms_level,
    preprocessing_fn,
    valid_charge,
    custom_fields,
    metadata_df,
    progress,
    shape,
):
    """Test parsing to DataFrame."""
    out = spectra_to_stream(
        real_mzml,
        batch_size=1,
        ms_level=ms_level,
        preprocessing_fn=preprocessing_fn,
        valid_charge=valid_charge,
        custom_fields=custom_fields,
        metadata_df=metadata_df,
        progress=progress,
    )
    parsed = pl.from_arrow(list(out))
    assert parsed.shape == shape
