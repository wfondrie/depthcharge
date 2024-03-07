"""Helper functions for testing."""

from typing import Any

import torch


def assert_dicts_equal(
    dict1: dict[Any], dict2: dict[Any], **kwargs: dict
) -> None:
    """Assert two dictionary are equal, while considering tensors.

    Parameters
    ----------
    dict1 : dict
        The first dictionary to compare.
    dict2 : dict
        The second dictionary to compare.
    **kwargs : dict
        Keyword arguments passed to `torch.testing.assert_close`

    Raises
    ------
    AssertionError
        Indicates that the two dictionaries are not equal.

    """
    bad_keys = []
    assert set(dict1.keys()) == set(dict2.keys())

    for key, val1 in dict1.items():
        try:
            val2 = dict2[key]
        except KeyError:
            bad_keys.append(key)
            continue

        try:
            assert type(val1) is type(val2)
        except AssertionError:
            bad_keys.append(key)
            continue

        try:
            assert val1 == val2
            continue
        except AssertionError:
            bad_keys.append(key)
            continue
        except RuntimeError:
            pass

        try:
            # Works on numpy arrays too.
            torch.testing.assert_close(val1, val2, **kwargs)
            continue
        except AssertionError:
            bad_keys.append(key)
            continue

    if not bad_keys:
        return

    raise AssertionError(
        f"Dictionaries did not match at the following keys: {bad_keys}"
    )
