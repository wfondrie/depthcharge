"""Ironically test that the testing functions are working."""

import numpy as np
import pytest
import torch

from depthcharge.testing import assert_dicts_equal


@pytest.mark.parametrize(
    ["dict1", "dict2", "error"],
    [
        ({"a": "a", "b": "b"}, {"a": "a", "b": "b"}, False),
        ({"a": "a"}, {"a": "c"}, True),
        ({"a": "a"}, {"b": "a"}, True),
        ({"a": torch.tensor([1])}, {"a": torch.tensor([1])}, False),
        ({"a": torch.tensor([1])}, {"a": torch.tensor([2])}, True),
        ({"a": torch.tensor([1])}, {"a": 1}, True),
        ({"a": np.array([1])}, {"a": np.array([1])}, False),
    ],
)
def test_assert_dicts_equal(dict1, dict2, error):
    """Test the dict equal function."""
    if error:
        with pytest.raises(AssertionError):
            assert_dicts_equal(dict1, dict2)
    else:
        assert_dicts_equal(dict1, dict2)
