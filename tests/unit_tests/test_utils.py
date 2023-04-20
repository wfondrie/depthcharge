"""Test that the utility function work."""
import pytest


def test_no_tensorboard(monkeypatch):
    """Test that import works without tensorboard"""
    import sys

    sys.modules["tensorboard"] = None

    import depthcharge as dc

    with pytest.raises(ImportError):
        dc.utils.read_tensorboard_scalars(".")
