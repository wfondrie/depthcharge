"""Test the encoders."""

import warnings

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


def test_float_encoder_dtype_conversion():
    """Test FloatEncoder buffers stay at float32 after dtype conversion."""
    enc_bf16 = FloatEncoder(8, 0.1, 10).bfloat16()
    X = torch.tensor([[0.0, 0.1, 10, 0.256]])

    # In the case of static wavelens, sin/cos_term will always be float32
    assert enc_bf16.sin_term.dtype == torch.float32
    assert enc_bf16.cos_term.dtype == torch.float32
    assert enc_bf16._dtype_tracker.dtype == torch.bfloat16
    Y = enc_bf16(X.bfloat16())
    assert Y.dtype == torch.bfloat16


def test_float_encoder_learnable_dtype_conversion():
    """Test learnable parameters follow dtype conversion."""
    enc_bf16 = FloatEncoder(8, 0.1, 10, learnable_wavelengths=True).bfloat16()
    X = torch.tensor([[0.0, 0.1, 10, 0.256]])

    # In the case of learnable wavelengths, sin/cos_term will be cast
    assert enc_bf16.sin_term.dtype == torch.bfloat16
    assert enc_bf16.cos_term.dtype == torch.bfloat16
    assert enc_bf16._dtype_tracker.dtype == torch.bfloat16
    Y = enc_bf16(X.bfloat16())
    assert Y.dtype == torch.bfloat16


def test_float_encoder_precision_warning():
    """Test warning is issued for lower precision inputs."""
    enc = FloatEncoder(8, 0.1, 10)
    X = torch.tensor([[0.0, 0.1, 10, 0.256]])

    # Should warn for bfloat16
    with pytest.warns(UserWarning, match="lower precision than float32"):
        enc(X.bfloat16())

    # Should warn for float16
    with pytest.warns(UserWarning, match="lower precision than float32"):
        enc(X.half())

    # Should not warn for float32
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enc(X.float())


def test_positional_encoder_dtype_mismatch_warning():
    """Test PositionalEncoder warns on dtype mismatch."""
    enc_bf16 = PositionalEncoder(8, 1, 8).bfloat16()
    X = torch.zeros(2, 5, 8)

    assert enc_bf16.sin_term.dtype == torch.float32

    with pytest.warns(UserWarning, match="does not match model dtype"):
        enc_bf16(X.float())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enc_bf16(X.bfloat16())


def test_float_encoder_numerical_stability():
    """Test that internal computation uses float32 for numerical stability."""
    enc = FloatEncoder(8, 0.001, 10000)

    # Use moderate values to test stability
    X = torch.tensor([[10.0, 50.0, 100.0]])

    Y_f32 = enc(X.float())

    enc_bf16 = enc.bfloat16()
    Y_bf16_f32_input = enc_bf16(X.float())

    torch.testing.assert_close(
        Y_f32,
        Y_bf16_f32_input.float(),
        rtol=1e-3,
        atol=1e-3,
    )
