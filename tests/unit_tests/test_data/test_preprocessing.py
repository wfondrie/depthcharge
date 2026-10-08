"""Test the preprocessing functions."""

import numpy as np

from depthcharge.data import preprocessing
from depthcharge.primitives import MassSpectrum


def test_scale_to_unit_norm():
    """Test that intensities are scaled to unit L2 norm."""
    spec = MassSpectrum(
        "test",
        "scan=1",
        np.array([100.0, 200.0]),
        np.array([3.0, 4.0]),
    )

    spec = preprocessing.scale_to_unit_norm(spec)
    np.testing.assert_allclose(spec.intensity, [0.6, 0.8])
    np.testing.assert_allclose(np.linalg.norm(spec.intensity), 1.0)
