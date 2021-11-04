"""A flexible feed-forward neural network."""
import torch
import numpy as np


class FeedForward(torch.nn.Module):
    """Create a feed forward neural net with leaky GELU activations

    Parameters
    ----------
    in_dim : int
        The input dimensionality.
    out_dim : int
        The output dimensionality.
    layers : int or tuple of int.
        If an int, layer sizes are linearly interpolated between the input and
        output dimensions using this number of layers. Otherwise, each element
        specifies the size of a layer.
    dropout : float, optional
        If greater than zero, add dropout layers with the specified
        probability.
    append : torch.nn.Module or None, optional
        A final layer to append, such as a sigmoid or tanh.
    """

    def __init__(self, in_dim, out_dim, layers, dropout=0, append=None):
        """Initiazlize a FeedForward network"""
        super().__init__()
        try:
            sizes = np.array([in_dim] + list(layers) + [out_dim])
        except TypeError:
            sizes = np.ceil(np.linspace(in_dim, out_dim, int(layers) + 1))

        sizes = sizes.astype(int)
        stack = []
        for idx in range(len(sizes) - 1):
            stack.append(torch.nn.Linear(sizes[idx], sizes[idx + 1]))
            if idx < len(sizes) - 2:
                stack.append(torch.nn.LeakyReLU())
                if dropout:
                    stack.append(torch.nn.Dropout(dropout))

        if append is not None:
            stack.append(append)

        self.layers = torch.nn.Sequential(*stack)

    def forward(self, X):
        """The forward pass"""
        return self.layers(X)
