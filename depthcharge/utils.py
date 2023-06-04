"""Common utility functions."""
from typing import Any


def listify(obj: Any) -> list[Any]:  # noqa: ANN401
    """Turn an object into a list, but don't split strings."""
    try:
        assert not isinstance(obj, str)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)


def check_int(integer: int, name: str) -> int:
    """Verify that an object is an integer, or coercible to one.

    Parameters
    ----------
    integer : int
        The integer to check.
    name : str
        The name to print in the error message if it fails.

    Returns
    -------
    int
        The coerced integer.
    """
    if isinstance(integer, int):
        return integer

    # Else if it is a float:
    coerced = int(integer)
    if coerced != integer:
        raise ValueError(f"'{name}' must be an integer.")

    return coerced


def check_positive_int(integer: int, name: str) -> int:
    """Verify that an object is an integer and positive.

    Parameters
    ----------
    integer : int
        The integer to check.
    name : str
        The name to print in the error message if it fails.

    Returns
    -------
    int
        The coerced integer.
    """
    try:
        integer = check_int(integer, name)
        assert integer > 0
    except (ValueError, AssertionError):
        raise ValueError(f"'{name}' must be a positive integer.")

    return integer
