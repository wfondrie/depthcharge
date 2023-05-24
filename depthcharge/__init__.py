"""Initialize the depthcharge package."""
from . import (
    data,
    encoders,
    feedforward,
    tokenizers,
    transformers,
)
from .primitives import (
    MassSpectrum,
    Molecule,
    Peptide,
    PeptideIons,
)
from .version import _get_version

__version__ = _get_version()
