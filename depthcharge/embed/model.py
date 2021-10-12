"""The depthcharge embedding model"""
import logging

import torch
import numpy as np
import pytorch_lightning as pl

from ..components import SpectrumEncoder, FeedForward, ModelMixin

LOGGER = logging.getLogger(__name__)


class SiameseSpectrumEncoder(pl.LightningModule, ModelMixin):
    """A Transformer model for self-supervised learning of spectrum embeddings.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimension for the spectum and each peak learned by the
        transformer.
    n_head : int, optional
        The number of attention heads in each layer. `d_model` must be divisble
        by `n_head`.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the
        transformer layers of the model.
    n_layers : int, optional
        The number of transformer layers.
    n_head_layer : int or tuple of int, optional
        The number of hidden layers for the multilayer perceptron head.
        Alternatively this may be a tuple of integers specifying the size of
        each linear layer.
    dropout : float, optional
        The dropout rate in all layers of the model.
    **kwargs : Dict
        Keyword arguments passed to the Adam optimizer
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        n_head_layers=3,
        dropout=0,
        n_log=10,
        **kwargs,
    ):
        """Initialize a SpectrumTransformer"""
        super().__init__()

        # Writable:
        self.n_log = n_log

        # Build the model:
        self.encoder = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.head = FeedForward(
            in_dim=2 * dim_model,
            out_dim=1,
            layers=n_head_layers,
            append=torch.nn.Sigmoid(),
        )

        self.mse = torch.nn.MSELoss()
        self.opt_kwargs = kwargs
        self._history = []

    def forward(self, X):
        """The forward pass"""
        return self.encoder(X)[0][:, 0, :]

    def step(self, batch):
        """A single step during training."""
        X, Y, gsp = batch
        X_emb = self(X)
        Y_emb = self(Y)
        pred = self.head(torch.hstack([X_emb, Y_emb])).squeeze()
        return self.mse(pred, gsp)

    def training_step(self, batch, *args):
        """A single training step with the model."""
        loss = self.step(batch)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, *args):
        """A single validation step with the model."""
        loss = self.step(batch)
        self.log("valid_loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        """Log the epoch metrics to self.history"""
        metrics = {"epoch": self.trainer.current_epoch}
        metrics.update(
            {k: v.item() for k, v in self.trainer.callback_metrics.items()}
        )
        self._history.append(metrics)
        if len(self._history) == 1:
            LOGGER.info("---------------------------------------")
            LOGGER.info("  Epoch |   Train Loss  |  Valid Loss  ")
            LOGGER.info("---------------------------------------")

        if not metrics["epoch"] % self.n_log:
            LOGGER.info(
                "  %5i | %15.7f | %15.7f ",
                metrics["epoch"],
                metrics.get("train_loss", np.nan),
                metrics["valid_loss"],
            )

    def configure_optimizers(self):
        """Initialize our optimizer"""
        return torch.optim.Adam(self.parameters(), **self.opt_kwargs)