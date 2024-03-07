"""Helpful Mixins."""

import torch


class ModelMixin:
    """Add helpful properties for depthcharge models."""

    @property
    def device(self) -> torch.device:
        """The current device for first parameter of the model."""
        return next(self.parameters()).device


class TransformerMixin:
    """Properties shared by Transformer models."""

    @property
    def d_model(self) -> int:
        """The latent dimensionality of the model."""
        return self._d_model

    @property
    def nhead(self) -> int:
        """The number of attention heads."""
        return self._nhead

    @property
    def dim_feedforward(self) -> int:
        """The dimensionality of the Transformer feedforward layers."""
        return self._dim_feedforward

    @property
    def n_layers(self) -> int:
        """The number of Transformer layers."""
        return self._n_layers

    @property
    def dropout(self) -> float:
        """The dropout for the transformer layers."""
        return self._dropout
