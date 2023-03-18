"""Test the encoders"""
import torch

from depthcharge.components.encoders import PositionalEncoder


def test_positional_encoder():
    """Test the positional encoder"""
    enc = PositionalEncoder(4.0, 100)
    X = torch.tensor([[[0.0, 100, 0, 100]]])
    Y = torch.tensor([[[0.0, 0, 1, 1]]])
    Y_hat = enc(X)
    torch.testing.assert_allclose(Y, Y_hat)
