"""Test getting the version."""

from importlib.metadata import PackageNotFoundError

import depthcharge


def test_version():
    """Just make sure its something."""
    import depthcharge

    assert depthcharge.__version__ is not None
    assert depthcharge.__version__


def test_bad_version(monkeypatch):
    """Test for a bad version."""

    def bad_version(*args, **kwargs):
        """A bad version."""
        raise PackageNotFoundError

    monkeypatch.setattr("depthcharge.version.version", bad_version)
    assert depthcharge.version._get_version() is None
