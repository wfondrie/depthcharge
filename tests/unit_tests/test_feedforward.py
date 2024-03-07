"""Test the feedforward model."""

import torch

from depthcharge.feedforward import FeedForward


def test_init():
    """Test initialization."""
    X = torch.randn(20, 3, 10)

    model = FeedForward(10, 1, 2)
    assert isinstance(model.layers[0], torch.nn.Linear)
    assert isinstance(model.layers[1], torch.nn.LeakyReLU)
    assert isinstance(model.layers[2], torch.nn.Linear)
    assert model.layers[0].in_features == 10
    assert model.layers[0].out_features == 6
    assert model.layers[2].in_features == 6
    assert model.layers[2].out_features == 1
    assert len(model.layers) == 3
    Y = model(X)
    assert Y.shape == (20, 3, 1)

    model = FeedForward(10, 1, 2, append=torch.nn.GELU())
    assert len(model.layers) == 4
    assert isinstance(model.layers[-1], torch.nn.GELU)
    Y = model(X)
    assert Y.shape == (20, 3, 1)

    model = FeedForward(10, 1, 2, dropout=0.1)
    assert len(model.layers) == 4
    assert isinstance(model.layers[2], torch.nn.Dropout)
    assert model.layers[2].p == 0.1
    Y = model(X)
    assert Y.shape == (20, 3, 1)

    model = FeedForward(10, 1, (50, 3))
    assert isinstance(model.layers[0], torch.nn.Linear)
    assert isinstance(model.layers[1], torch.nn.LeakyReLU)
    assert isinstance(model.layers[2], torch.nn.Linear)
    assert isinstance(model.layers[3], torch.nn.LeakyReLU)
    assert isinstance(model.layers[4], torch.nn.Linear)
    assert model.layers[0].in_features == 10
    assert model.layers[0].out_features == 50
    assert model.layers[2].in_features == 50
    assert model.layers[2].out_features == 3
    assert model.layers[4].in_features == 3
    assert model.layers[4].out_features == 1
    assert len(model.layers) == 5
    Y = model(X)
    assert Y.shape == (20, 3, 1)

    model = FeedForward(10, 1, 2, activation=torch.nn.GELU())
    assert len(model.layers) == 3
    assert isinstance(model.layers[1], torch.nn.GELU)
    Y = model(X)
    assert Y.shape == (20, 3, 1)
