"""Test that parsers work."""

import shutil

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from depthcharge.data import CustomField, hash_peak_file
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
            "peak_file_hash": [hash_peak_file(mgf_small)] * 2,
            "scan_id": ["index=0", "index=1"],
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
            pl.col("intensity_array").cast(pl.List(pl.Float32)),
            pl.col("ms_level").cast(pl.UInt8),
            pl.col("precursor_charge").cast(pl.Int16),
        ]
    )

    assert parsed.shape == (2, 8)
    assert_frame_equal(parsed, expected)

    parsed = pl.from_arrow(
        MgfParser(mgf_small, valid_charge=[2]).iter_batches(2),
    )
    assert parsed.shape == (1, 8)
    assert isinstance(ParserFactory.get_parser(mgf_small), MgfParser)


@pytest.mark.parametrize(
    ["ms_level", "preprocessing_fn", "valid_charge", "custom_fields", "shape"],
    [
        (2, None, None, None, (4, 8)),
        (1, None, None, None, (4, 8)),
        (3, None, None, None, (3, 8)),
        (2, None, [3], None, (3, 8)),
        (None, None, None, None, (11, 8)),
        (2, scale_to_unit_norm, None, MZML_FIELD, (4, 9)),
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
        (2, None, None, None, (4, 8)),
        (1, None, None, None, (4, 8)),
        (3, None, None, None, (3, 8)),
        (2, None, [3], None, (3, 8)),
        (None, None, None, None, (11, 8)),
        (2, scale_to_unit_norm, None, MZXML_FIELD, (4, 9)),
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
        (2, None, None, None, (7, 8)),
        (1, None, None, None, (7, 8)),
        (3, None, None, None, (7, 8)),
        (2, None, [3], None, (3, 8)),
        (None, None, None, None, (7, 8)),
        (2, scale_to_unit_norm, None, MGF_FIELD, (7, 9)),
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
        (2, None, None, None, (3, 8)),
        (2, None, [3], None, (1, 8)),
        (None, None, None, None, (3, 8)),
        (None, scale_to_unit_norm, None, None, (3, 8)),
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


def test_skipped_spectra_warning(mgf_small):
    """Test that invalid custom fields are skipped with a warning."""
    parser = MgfParser(
        mgf_small,
        custom_fields=CustomField(
            "seq", lambda x: x["params"]["bar"], pa.string()
        ),
    )

    msg = r"^Skipped 2 spectra with invalid information\. Last error was: "
    with pytest.warns(UserWarning, match=msg + r"KeyError\('bar'\)$"):
        assert not list(parser.iter_batches(None))

    # The warning is still raised if iteration stops early:
    def accessor(spectrum: dict) -> str:
        seq = spectrum["params"]["seq"]
        if seq == "LESLIEK":
            raise ValueError("bad seq")

        return seq

    parser = MgfParser(
        mgf_small,
        custom_fields=CustomField("seq", accessor, pa.string()),
    )

    batches = parser.iter_batches(1)
    assert next(batches)["seq"].to_pylist() == ["EDITHR"]
    with pytest.warns(UserWarning, match=msg.replace("2", "1")):
        batches.close()


def test_invalid_file(tmp_path):
    """Test an invalid file raises an error."""
    tmp_path.touch("blah.txt")

    with pytest.raises(OSError):
        ParserFactory().get_parser(tmp_path / "blah.txt")


def test_hash_peak_file(mgf_small, tmp_path):
    """Test peak file fingerprints."""
    expected = hash_peak_file(mgf_small)
    assert len(expected) == 32

    # A copy in another directory has the same hash:
    copied = tmp_path / "copy" / mgf_small.name
    copied.parent.mkdir()
    shutil.copy(mgf_small, copied)
    assert hash_peak_file(copied) == expected

    # A file with the same name but different contents does not:
    modified = tmp_path / "modified" / mgf_small.name
    modified.parent.mkdir()
    modified.write_text(mgf_small.read_text() + "\n")
    assert hash_peak_file(modified) != expected


def test_hash_large_file(tmp_path):
    """Test that only the start, end, and size of large files are used."""
    n_bytes = 1024
    data = bytearray(range(256)) * 16  # 4096 bytes

    def write_hash(name, contents):
        path = tmp_path / name
        path.write_bytes(bytes(contents))
        return hash_peak_file(path, n_bytes=n_bytes)

    original = write_hash("original", data)

    changed_end = data.copy()
    changed_end[-1] ^= 1
    assert write_hash("end", changed_end) != original

    changed_start = data.copy()
    changed_start[0] ^= 1
    assert write_hash("start", changed_start) != original

    # The middle is not read:
    changed_middle = data.copy()
    changed_middle[2048] ^= 1
    assert write_hash("middle", changed_middle) == original

    assert write_hash("longer", data + b"\0") != original


def test_hash_directory(real_tdf, tmp_path):
    """Test fingerprints of directories, such as Bruker .d."""
    expected = hash_peak_file(real_tdf)

    copied = tmp_path / real_tdf.name
    shutil.copytree(real_tdf, copied)
    assert hash_peak_file(copied) == expected

    (copied / "extra.txt").write_text("extra")
    assert hash_peak_file(copied) != expected
