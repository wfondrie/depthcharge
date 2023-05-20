"""Common utility functions."""
from os import PathLike
from typing import Any

try:
    import pandas as pd
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError:
    EventAccumulator = None
    pd = None


def read_tensorboard_scalars(path: PathLike) -> pd.DataFrame:
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
    if EventAccumulator is None:
        raise ImportError(
            "Install the 'tensorboard' optional dependency to "
            "use this function. https://www.tensorflow.org/tensorboard"
        )

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


def listify(obj: Any) -> list[Any]: # noqa: ANN401
    """Turn an object into a list, but don't split strings."""
    try:
        assert not isinstance(obj, str)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)
