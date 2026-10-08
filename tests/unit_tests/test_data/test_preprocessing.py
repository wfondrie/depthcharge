"""Test the preprocessing functions."""

import numpy as np
import pytest

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


@pytest.mark.parametrize("n_peaks", [0, 1, 50, 200, 201, 1000])
@pytest.mark.parametrize("below_min_mz", [False, True])
@pytest.mark.parametrize("zeros", [False, True])
def test_default(n_peaks, below_min_mz, zeros):
    """Test that the default preprocessing matches spectrum_utils."""
    rng = np.random.default_rng(n_peaks)
    max_mz = 139 if below_min_mz else 2000
    mz = rng.uniform(50, max_mz, n_peaks)
    intensity = rng.uniform(0, 1e5, n_peaks)
    if zeros:
        intensity[: n_peaks // 2] = 0

    def spectrum() -> MassSpectrum:
        """Create the mass spectrum.

        Returns
        -------
        MassSpectrum
            The unprocessed mass spectrum.

        """
        return MassSpectrum("test", "scan=1", mz.copy(), intensity.copy())

    expected = spectrum()
    for func in [
        preprocessing.set_mz_range(min_mz=140),
        preprocessing.filter_intensity(max_num_peaks=200),
        preprocessing.scale_intensity(scaling="root"),
        preprocessing.scale_to_unit_norm,
    ]:
        expected = func(expected)

    result = preprocessing._default(spectrum())
    np.testing.assert_array_equal(result.mz, expected.mz)
    np.testing.assert_array_equal(result.intensity, expected.intensity)
    assert result.intensity.dtype == np.float32
