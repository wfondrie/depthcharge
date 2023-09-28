"""Test the arrow functionality."""
import polars as pl
import pytest

from depthcharge.data.arrow import (
    spectra_to_df,
    spectra_to_parquet,
    spectra_to_stream,
)
from depthcharge.data.preprocessing import scale_to_unit_norm


@pytest.mark.parametrize(
    ["ms_level", "preprocessing_fn", "valid_charge", "custom_fields", "shape"],
    [
        (2, None, None, None, (4, 7)),
        (1, None, None, None, (4, 7)),
        (3, None, None, None, (3, 7)),
        (2, None, [3], None, (3, 7)),
        (None, None, None, None, (11, 7)),
        (2, scale_to_unit_norm, None, {"index": ["index"]}, (4, 8)),
    ],
)
def test_to_df(
    real_mzml, ms_level, preprocessing_fn, valid_charge, custom_fields, shape
):
    """Test parsing to DataFrame."""
    parsed = spectra_to_df(
        real_mzml,
        ms_level=ms_level,
        preprocessing_fn=preprocessing_fn,
        valid_charge=valid_charge,
        custom_fields=custom_fields,
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
        (2, scale_to_unit_norm, None, {"index": ["index"]}, (4, 8)),
    ],
)
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


@pytest.mark.parametrize(
    ["ms_level", "preprocessing_fn", "valid_charge", "custom_fields", "shape"],
    [
        (2, None, None, None, (4, 7)),
        (1, None, None, None, (4, 7)),
        (3, None, None, None, (3, 7)),
        (2, None, [3], None, (3, 7)),
        (None, None, None, None, (11, 7)),
        (2, scale_to_unit_norm, None, {"index": ["index"]}, (4, 8)),
    ],
)
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
