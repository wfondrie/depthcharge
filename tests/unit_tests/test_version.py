"""Test getting the version"""
import depthcharge


def test_version():
    """Just make sure its something."""
    assert depthcharge.__version__ is not None
    assert depthcharge.__version__
