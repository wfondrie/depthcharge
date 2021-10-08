"""Useful mixins for model classes"""
import pandas as pd


class ModelMixin:
    """Add some useful methods for models."""

    def init(self):
        """Initialize the ModelMixin"""
        self._history = []

    @property
    def history(self):
        """The training history of a model."""
        return pd.DataFrame(self._history)

    @property
    def n_parameters(self):
        """The number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
