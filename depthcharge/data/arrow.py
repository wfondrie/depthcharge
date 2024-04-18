"""Store spectrum data as Arrow tables."""

from collections.abc import Callable, Generator, Iterable
from os import PathLike
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from cloudpathlib import AnyPath

from .fields import CustomField
from .parsers import ParserFactory


def spectra_to_stream(
    peak_file: PathLike,
    *,
    batch_size: int | None = 100_000,
    metadata_df: pl.DataFrame | pl.LazyFrame | None = None,
    ms_level: int | Iterable[int] | None = 2,
    preprocessing_fn: Callable | Iterable[Callable] | None = None,
    valid_charge: Iterable[int] | None = None,
    custom_fields: CustomField | Iterable[CustomField] | None = None,
    progress: bool = True,
) -> Generator[pa.RecordBatch]:
    """Stream mass spectra in an Apache Arrow format, with preprocessing.

    Apache Arrow is a space efficient, columnar data format
    that is popular in the data science and engineering community.
    This function reads data from a mass spectrometry data format,
    extracts the mass spectrum and identifying information. By
    default, the schema is:
        peak_file: str
        scan_id: int
        ms_level: int
        precursor_mz: float
        precursor_charge: int
        mz_array: list[float]
        intensity_array: list[float]

    An optional metadata DataFrame can be provided to add additional
    metadata to each mass spectrum. This DataFrame must contain
    a ``scan_id`` column containing the integer scan identifier for
    each mass spectrum. For mzML files, this is generally the integer
    following ``scan=``, whereas for MGF files this is the zero-indexed
    offset of the mass spectrum in the file.

    Finally, custom fields can be extracted from the mass spectrometry
    data file for advanced use. This must be a CustomField, where the
    name is the new column and the accessor is a function
    to extract a value from the corresponding Pyteomics spectrum
    dictionary. The pyarrow data type must also be specified.

    Parameters
    ----------
    peak_file : PathLike, optional
        The mass spectrometry data file in mzML, mzXML, or MGF format.
    batch_size : int or None
        The number of mass spectra in each RecordBatch. ``None`` will load
        all of the spectra in a single batch.
    metadata_df : polars.DataFrame or polars.LazyFrame, optional
        A `polars.DataFrame` containing additional
        metadata from the spectra. This is merged on the `scan_id` column
        which must be present, and optionally a `peak_file` column,
        if present.
    ms_level : int, list of int, or None, optional
        The level(s) of tandem mass spectra to keep. `None` will retain
        all spectra.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra. `None`,
        the default, filters for the top 200 peaks above m/z 140,
        square root transforms the intensities and scales them to unit norm.
        See the preprocessing module for details and additional options.
    valid_charge : int or list of int, optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    custom_fields : CustomField or iterable of CustomField
        Additional fields to extract during peak file parsing.
    progress : bool, optional
        Enable or disable the progress bar.

    Returns
    -------
    Generator of pyarrow.RecordBatch
        Batches of parsed spectra.

    """
    parser_args = {
        "ms_level": ms_level,
        "valid_charge": valid_charge,
        "preprocessing_fn": preprocessing_fn,
        "custom_fields": custom_fields,
        "progress": progress,
    }

    on_cols = ["scan_id"]
    validation = "1:1"
    if metadata_df is not None:
        metadata_df = metadata_df.lazy()
        if "peak_file" in metadata_df.columns:
            # Validation is only supported when on is a single column.
            # Adding a footgun here to remove later...
            validation = "m:m"
            on_cols.append("peak_file")

    parser = ParserFactory.get_parser(peak_file, **parser_args)
    for batch in parser.iter_batches(batch_size=batch_size):
        if metadata_df is not None:
            batch = (
                pl.from_arrow(batch)
                .lazy()
                .join(
                    metadata_df,
                    on=on_cols,
                    how="left",
                    validate=validation,
                )
                .collect()
                .to_arrow()
                .to_batches(max_chunksize=batch_size)[0]
            )

        yield batch


def spectra_to_parquet(
    peak_file: PathLike,
    *,
    parquet_file: PathLike = None,
    batch_size: int = 100_000,
    metadata_df: pl.DataFrame | None = None,
    ms_level: int | Iterable[int] | None = 2,
    preprocessing_fn: Callable | Iterable[Callable] | None = None,
    valid_charge: Iterable[int] | None = None,
    custom_fields: CustomField | Iterable[CustomField] | None = None,
    progress: bool = True,
) -> Path:
    """Stream mass spectra to Apache Parquet, with preprocessing.

    Apache Parquet is a space efficient, columnar data storage format
    that is popular in the data science and engineering community.
    This function reads data from a mass spectrometry data format,
    extracts the mass spectrum and identifying information. By
    default, the schema is:
        peak_file: str
        scan_id: int
        ms_level: int
        precursor_mz: float64
        precursor_charge: int8
        mz_array: list[float64]
        intensity_array: list[float64]

    An optional metadata DataFrame can be provided to add additional
    metadata to each mass spectrum. This DataFrame must contain
    a ``scan_id`` column containing the integer scan identifier for
    each mass spectrum. For mzML files, this is generally the integer
    following ``scan=``, whereas for MGF files this is the zero-indexed
    offset of the mass spectrum in the file.

    Finally, custom fields can be extracted from the mass spectrometry
    data file for advanced use. This must be a CustomField, where the
    name is the new column and the accessor is a function
    to extract a value from the corresponding Pyteomics spectrum
    dictionary. The pyarrow data type must also be specified.

    Parameters
    ----------
    peak_file : PathLike, optional
        The mass spectrometry data file in mzML, mzXML, or MGF format.
    parquet_file : PathLike, optional
        The output file. By default this is the input file stem with a
        `.parquet` extension.
    batch_size : int
        The number of mass spectra to process simultaneously.
    metadata_df : polars.DataFrame or polars.LazyFrame, optional
        A `polars.DataFrame` containing additional
        metadata from the spectra. This is merged on the `scan_id` column
        which must be present, and optionally a `peak_file` column,
        if present.
    ms_level : int, list of int, or None, optional
        The level(s) of tandem mass spectra to keep. `None` will retain
        all spectra.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra. `None`,
        the default, filters for the top 200 peaks above m/z 140,
        square root transforms the intensities and scales them to unit norm.
        See the preprocessing module for details and additional options.
    valid_charge : int or list of int, optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    custom_fields : CustomField or iterable of CustomField
        Additional fields to extract during peak file parsing.
    progress : bool, optional
        Enable or disable the progress bar.

    Returns
    -------
    Path
        The Parquet file that was written.

    """
    streamer = spectra_to_stream(
        peak_file=peak_file,
        batch_size=batch_size,
        metadata_df=metadata_df,
        ms_level=ms_level,
        preprocessing_fn=preprocessing_fn,
        valid_charge=valid_charge,
        custom_fields=custom_fields,
        progress=progress,
    )

    if parquet_file is None:
        parquet_file = Path(AnyPath(peak_file).stem).with_suffix(".parquet")

    try:
        writer = None
        for batch in streamer:
            if writer is None:
                writer = pq.ParquetWriter(parquet_file, schema=batch.schema)

            writer.write_batch(batch)
    finally:
        writer.close()

    return parquet_file


def spectra_to_df(
    peak_file: PathLike,
    *,
    metadata_df: pl.DataFrame | None = None,
    ms_level: int | Iterable[int] | None = 2,
    preprocessing_fn: Callable | Iterable[Callable] | None = None,
    valid_charge: Iterable[int] | None = None,
    custom_fields: CustomField | Iterable[CustomField] | None = None,
    progress: bool = True,
) -> pl.DataFrame:
    """Read mass spectra into a Polars DataFrame.

    Apache Parquet is a space efficient, columnar data storage format
    that is popular in the data science and engineering community.
    This function reads data from a mass spectrometry data format,
    extracts the mass spectrum and identifying information. By
    default, the schema is:
        peak_file: str
        scan_id: int
        ms_level: int
        precursor_mz: float64
        precursor_charge: int8
        mz_array: list[float64]
        intensity_array: list[float64]

    An optional metadata DataFrame can be provided to add additional
    metadata to each mass spectrum. This DataFrame must contain
    a ``scan_id`` column containing the integer scan identifier for
    each mass spectrum. For mzML files, this is generally the integer
    following ``scan=``, whereas for MGF files this is the zero-indexed
    offset of the mass spectrum in the file.

    Finally, custom fields can be extracted from the mass spectrometry
    data file for advanced use. This must be a CustomField, where the
    name is the new column and the accessor is a function
    to extract a value from the corresponding Pyteomics spectrum
    dictionary. The pyarrow data type must also be specified.

    Parameters
    ----------
    peak_file : PathLike, optional
        The mass spectrometry data file in mzML, mzXML, or MGF format.
    metadata_df : polars.DataFrame or polars.LazyFrame, optional
        A `polars.DataFrame` containing additional
        metadata from the spectra. This is merged on the `scan_id` column
        which must be present, and optionally a `peak_file` column,
        if present.
    ms_level : int, list of int, or None, optional
        The level(s) of tandem mass spectra to keep. `None` will retain
        all spectra.
    preprocessing_fn : Callable or Iterable[Callable], optional
        The function(s) used to preprocess the mass spectra. `None`,
        the default, filters for the top 200 peaks above m/z 140,
        square root transforms the intensities and scales them to unit norm.
        See the preprocessing module for details and additional options.
    valid_charge : int or list of int, optional
        Only consider spectra with the specified precursor charges. If `None`,
        any precursor charge is accepted.
    custom_fields : CustomField or iterable of CustomField
        Additional fields to extract during peak file parsing.
    progress : bool, optional
        Enable or disable the progress bar.

    Returns
    -------
    polars.DataFrame
        A dataframe containing the parsed mass spectra.

    """
    streamer = spectra_to_stream(
        peak_file=peak_file,
        batch_size=None,
        metadata_df=metadata_df,
        ms_level=ms_level,
        preprocessing_fn=preprocessing_fn,
        valid_charge=valid_charge,
        custom_fields=custom_fields,
        progress=progress,
    )

    return pl.from_arrow(streamer)
