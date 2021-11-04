"""Common utility functions"""
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def read_tensorboard_scalars(path):
    """Read scalars from Tensorboard logs.

    Parameters
    ----------
    path : str
        The path of the scalar log file.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the scalar values.
    """
    event = EventAccumulator(path)
    event.Reload()
    data = []
    for tag in event.Tags()["scalars"]:
        tag_df = pd.DataFrame(
            event.Scalars(tag), columns=["wall_time", "step", "value"]
        )
        tag_df["tag"] = tag
        data.append(tag_df)

    return pd.concat(data)


def listify(obj):
    """Turn an object into a list, but don't split strings."""
    try:
        assert not isinstance(obj, str)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)


# For Parameter Checking ------------------------------------------------------
def check_int(integer, name):
    """Verify that an object is an integer, or coercible to one.

    Parameters
    ----------
    integer : int
        The integer to check.
    name : str
        The name to print in the error message if it fails.
    """
    if isinstance(integer, int):
        return integer

    # Else if it is a float:
    coerced = int(integer)
    if coerced != integer:
        raise ValueError(f"'{name}' must be an integer.")

    return coerced


def check_positive_int(integer, name):
    """Verify that an object is an integer and positive.

    Parameters
    ----------
    integer : int
        The integer to check.
    name : str
        The name to print in the error message if it fails.
    """
    try:
        integer = check_int(integer, name)
        assert integer > 0
    except (ValueError, AssertionError):
        raise ValueError(f"'{name}' must be a positive integer.")

    return integer
