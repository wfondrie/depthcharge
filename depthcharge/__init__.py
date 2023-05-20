"""Initialize the depthcharge package"""

from . import (
    data,
    transformers,
    encoders,
    base,
    mixins,
    utils,
)

from .version import _get_version

__version__ = _get_version()
