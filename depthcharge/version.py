"""Get the package version"""
from importlib.metadata import version, PackageNotFoundError


def _get_version():
    """Retreive the package version.

    Using setuptools-scm dynamically sets the pacakge version depending on
    the latest git release and commits since that point. This function
    returns the current version number.

    Returns
    -------
    str or None
        The package version number. If not version is found, returns None.
    """
    try:
        return version("depthcharge")
    except PackageNotFoundError:
        return None
