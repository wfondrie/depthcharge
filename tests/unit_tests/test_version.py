"""Test getting the version"""
import depthcharge
from importlib.metadata import PackageNotFoundError


def test_version():
    """Just make sure its something."""
    import depthcharge

    assert depthcharge.__version__ is not None
    assert depthcharge.__version__


def test_bad_version(monkeypatch):
    def bad_version(*args, **kwargs):
        raise PackageNotFoundError

    monkeypatch.setattr("depthcharge.version.version", bad_version)
    assert depthcharge.version._get_version() is None
