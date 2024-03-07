"""Initialize the depthcharge package."""

# Ignore a bunch of pkg_resources warnings from dependencies:
import warnings

with warnings.catch_warnings():
    for module in ["psims", "pkg_resources"]:
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=module,
        )

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
