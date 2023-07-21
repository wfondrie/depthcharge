"""A flexible feed-forward neural network."""
from collections.abc import Iterable

import numpy as np
import torch


class FeedForward(torch.nn.Module):
    """Create a feed forward neural network.

    Parameters
    ----------
    in_features : int
        The input dimensionality.
    out_features : int
        The output dimensionality.
    layers : int or tuple of int.
        If an int, layer sizes are linearly interpolated between the input and
        output dimensions using this number of layers. Otherwise, each element
        specifies the size of a layer.
    dropout : float, optionalf
        If greater than zero, add dropout layers with the specified
        probability.
    activation: torch.nn.Module, optional
        The activation function to place between layers.
    append : torch.nn.Module or None, optional
        A final layer to append, such as a sigmoid or tanh.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layers: int | Iterable[int],
        dropout: float = 0,
        activation: torch.nn.Module = torch.nn.LeakyReLU(),
        append: torch.nn.Module | None = None,
    ) -> None:
        """Initiazlize a FeedForward network."""
        super().__init__()
        try:
            sizes = np.array(
                [in_features] + list(layers) + [out_features],
            )
        except TypeError:
            sizes = np.ceil(
                np.linspace(in_features, out_features, int(layers) + 1),
            )

        sizes = sizes.astype(int)
        stack = []
        for idx in range(len(sizes) - 1):
            stack.append(torch.nn.Linear(sizes[idx], sizes[idx + 1]))
            if idx < len(sizes) - 2:
                stack.append(activation)
                if dropout:
                    stack.append(torch.nn.Dropout(dropout))

        if append is not None:
            stack.append(append)

        self.layers = torch.nn.Sequential(*stack)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The forward pass.

        Parameters
        ----------
        X : torch.Tensor of shape (..., in_features)
            The input tensor.

        Returns
        -------
        torch.Tensor of shape (..., out_features)
            The output tensor.
        """
        return self.layers(X)
