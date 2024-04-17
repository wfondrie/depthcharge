"""Test the encoders."""

import numpy as np
import pytest
import torch

from depthcharge.encoders import (
    FloatEncoder,
    PeakEncoder,
    PositionalEncoder,
)


def test_positional_encoder():
    """Test the positional encoder."""
    enc = PositionalEncoder(8, 1, 8)
    X = torch.zeros(1, 9, 8)
    Y = enc(X)

    period = torch.cat([torch.zeros(4), torch.ones(4)], axis=0)
    torch.testing.assert_close(Y[0, 0, :], period)
    torch.testing.assert_close(Y[0, 8, :], period)

    cycle = 2 * np.pi
    periods = torch.tensor(
        [
            1 / cycle,
            8 ** (1 / 3) / cycle,
            8 ** (2 / 3) / cycle,
            8 ** (3 / 3) / cycle,
        ]
    )

    expected_2 = torch.cat(
        [torch.sin(2 / periods), torch.cos(2 / periods)],
        axis=0,
    )
    torch.testing.assert_close(Y[0, 2, :], expected_2)

    X2 = torch.ones(1, 9, 8)
    Y2 = enc(X2)
    torch.testing.assert_close(Y2[0, 0, :], period + 1)


def test_float_encoder():
    """Test the float encodings."""
    enc = FloatEncoder(8, 0.1, 10)
    X = torch.tensor([[0, 0.1, 10, 0.256]])
    Y = enc(X)
    period = torch.cat([torch.zeros(4), torch.ones(4)], axis=0)
    torch.testing.assert_close(Y[0, 0, :], period)
    torch.testing.assert_close(Y[0, 1, (0, 4)], torch.tensor([0.0, 1]))
    torch.testing.assert_close(Y[0, 2, (3, 7)], torch.tensor([0.0, 1]))

    # Check for things in-between the expected period:
    assert Y[0, 3, :].min() > -0.99
    assert Y[0, 3, :].max() < 0.99

    enc = FloatEncoder(8, 0.1, 10, True)
    X = torch.tensor([[0, 0.1, 10, 0.256]])
    Y = enc(X)
    period = torch.cat([torch.zeros(4), torch.ones(4)], axis=0)
    torch.testing.assert_close(Y[0, 0, :], period)
    torch.testing.assert_close(Y[0, 1, (0, 4)], torch.tensor([0.0, 1]))
    torch.testing.assert_close(Y[0, 2, (3, 7)], torch.tensor([0.0, 1]))

    # Check for things in-between the expected period:
    assert Y[0, 3, :].min() > -0.99
    assert Y[0, 3, :].max() < 0.99
    assert isinstance(enc.sin_term, torch.nn.Parameter)


def test_invalid_float_encoder():
    """Test that errors are raised."""
    with pytest.raises(ValueError):
        FloatEncoder(8, 0, 10)

    with pytest.raises(ValueError):
        FloatEncoder(8, 10, 0)


def test_both_sinusoid():
    """Test that both encoders are sinusoidal."""
    enc = PeakEncoder(8)
    X = torch.tensor([[[0.0, 0], [0, 1]]])
    Y = enc(X)

    assert Y.shape == (1, 2, 8)
    assert isinstance(enc.int_encoder, FloatEncoder)

    enc = PeakEncoder(8, learnable_wavelengths=True)
    X = torch.tensor([[[0.0, 0], [0, 1]]])
    Y = enc(X)

    assert Y.shape == (1, 2, 8)
    assert isinstance(enc.int_encoder, FloatEncoder)
    assert isinstance(enc.int_encoder.sin_term, torch.nn.Parameter)
    assert isinstance(enc.mz_encoder.sin_term, torch.nn.Parameter)
