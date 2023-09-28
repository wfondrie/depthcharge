"""Common utility functions."""
from typing import Any

import polars as pl


def listify(obj: Any) -> list[Any]:  # noqa: ANN401
    """Turn an object into a list, but don't split strings."""
    try:
        assert not isinstance(obj, str)
        assert not isinstance(obj, pl.DataFrame)
        assert not isinstance(obj, pl.LazyFrame)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)
