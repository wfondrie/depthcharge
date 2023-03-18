"""Initialize the depthcharge package"""
from .version import _get_version
from . import components
from . import data
from . import utils

__version__ = _get_version()
