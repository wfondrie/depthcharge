"""Get the version information."""
from importlib.metadata import version, PackageNotFoundError


def _get_version() -> str:
    """Return the version information for deptcharge."""
    try:
        return version("depthcharge-ms")
    except PackageNotFoundError:
        return None
