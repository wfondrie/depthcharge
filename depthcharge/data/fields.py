"""Custom fields for the Arrow Schema."""

from collections.abc import Callable
from dataclasses import dataclass

import pyarrow as pa


@dataclass
class CustomField:
    """An additional field to extract during peak file parsing.

    The accessor function is used to extract additional information
    from each spectrum during parsing.

    Parameters
    ----------
    name: str
        The resulting column name in the Arrow schema.
    accessor: Callable
        A function to access the value of interest. The input will
        depend on the parser that is used. Currently, we use
        Pyteomics to parse mzML, MGF, and mzXML files, so the
        parameters for this function would be a dictionary for
        each spectrum.
    dtype: pyarrow.DataType
        The expected Arrow data type for the column in the schema.

    """

    name: str
    accessor: Callable
    dtype: pa.DataType
