"""Initialize the depthcharge package"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass


from . import embedder
from . import data
from . import denovo
