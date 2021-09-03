"""Initialize the depthcharge package"""
from .version import _get_version
from . import embedder
from . import data
from . import denovo

__version__ = _get_version()
