"""The depthcharge embedding model"""
import logging

import torch
import numpy as np
import pytorch_lightning as pl

from ..components import SpectrumEncoder, FeedForward, ModelMixin

LOGGER = logging.getLogger(__name__)


class SiameseSpectrumEncoder(pl.LightningModule, ModelMixin):
    """A Transformer model for self-supervised learning of mass spectrum
    embeddings.

    Use this model in conjuction with a pytorch-lightning Trainer.

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
    n_head_layers : int or tuple of int, optional
        The number of hidden layers for the multilayer perceptron head.
        Alternatively this may be a tuple of integers specifying the size of
        each linear layer.
    dropout : float, optional
        The dropout rate in all layers of the model.
    dim_intensity : int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value. If ``None``, the intensity will be projected
        up to ``dim_model`` using a linear layer, then summed with the m/z
        emcoding for each peak.
    n_log : int, optional
        The frequency with which to log losses to the console. These are only
        visible when the logging level is set to ``INFO``.
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
        dim_intensity=None,
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
            dim_intensity=dim_intensity,
        )

        self.head = FeedForward(
            in_dim=2 * dim_model,
            out_dim=1,
            layers=n_head_layers,
            # append=torch.nn.Sigmoid(),
            append=None,
        )

        self.mse = torch.nn.MSELoss()
        # self.mse = ZeroWeightedMSELoss(0.001)
        self.opt_kwargs = kwargs
        self._history = []

    def forward(self, X):
        """Embed a mass spectrum.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_peaks, 2)
            The mass spectra to embed, where ``X[:, :, 0]`` are the m/z values
            and ``X[:, :, 1]`` are their associated intensities.

        Returns
        -------
        torch.Tensor of shape (batch_size, dim_model)
            The embeddings for the mass spectra.
        """
        return self.encoder(X)[0][:, 0, :]

    def _step(self, batch):
        """Predict the GSP, without clipping, for two mass spectra.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain pairs of mass spectra
            (index 0 and 1) and the calcualted GSP between them (index 2).

        Returns
        -------
        predicted : torch.tensor of shape (batch_size,)
            The predicted GSP between the batch of mass spectra.
        truth : torch.tensor of shape (batch_size,)
            The calcualted GSP between the batch of mass spectra.
        """
        X, Y, gsp = batch
        X_emb = self(X)
        Y_emb = self(Y)
        return self.head(torch.hstack([X_emb, Y_emb])).squeeze(), gsp

    def predict_step(self, batch, *args):
        """Predict the GSP between two mass spectra.

        Note that this is used within the context of a pytorch-lightning
        Trainer to generate a prediction.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain pairs of mass spectra
            (index 0 and 1) and the calcualted GSP between them (index 2).

        Returns
        -------
        predicted : torch.tensor of shape (batch_size,)
            The predicted GSP between the batch of mass spectra.
        truth : torch.tensor of shape (batch_size,)
            The calcualted GSP between the batch of mass spectra.
        """
        pred, gsp = self._step(batch)
        return torch.clamp(pred, 0, 1), gsp

    def training_step(self, batch, *args):
        """A single training step with the model.

        Note that this is used within the context of a pytorch-lightning
        Trainer for each step in the training loop.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain pairs of mass spectra
            (index 0 and 1) and the calcualted GSP between them (index 2).

        Returns
        -------
        torch.tensor of shape (1)
            The mean squared error for the batch.
        """
        pred, gsp = self._step(batch)
        loss = self.mse(pred, gsp)
        self.log(
            "MSE",
            {"train": loss.item()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        with torch.no_grad():
            clipped = self.mse(torch.clamp(pred, 0, 1), gsp)

        self.log(
            "clipped_MSE",
            {"train": clipped.item()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, *args):
        """A single validation step with the model.

        Note that this is used within the context of a pytorch-lightning
        Trainer for each step in the validation loop.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain pairs of mass spectra
            (index 0 and 1) and the calcualted GSP between them (index 2).

        Returns
        -------
        torch.tensor of shape (1)
            The mean squared error for the batch.
        """
        pred, gsp = self._step(batch)
        loss = self.mse(pred, gsp)
        self.log(
            "MSE",
            {"valid": loss.item()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        with torch.no_grad():
            clipped = self.mse(torch.clamp(pred, 0, 1), gsp)

        self.log(
            "clipped_MSE",
            {"valid": clipped.item()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        """Log the training loss.

        This is a pytorch-lightning hook.
        """
        mse = "clipped_MSE"
        # mse = "MSE"
        metrics = {
            "epoch": self.trainer.current_epoch,
            "train": self.trainer.callback_metrics[mse]["train"].item(),
        }
        self._history.append(metrics)

    def on_validation_epoch_end(self):
        """Log the epoch metrics to self.history.

        This is a pytorch-lightning hook.
        """
        if not self._history:
            return

        mse = "clipped_MSE"
        # mse = "MSE"
        valid_loss = self.trainer.callback_metrics[mse]["valid"].item()
        self._history[-1]["valid"] = valid_loss
        if len(self._history) == 1:
            LOGGER.info("---------------------------------------")
            LOGGER.info("  Epoch |   Train Loss  |  Valid Loss  ")
            LOGGER.info("---------------------------------------")

        metrics = self._history[-1]
        if not metrics["epoch"] % self.n_log:
            LOGGER.info(
                "  %5i | %15.7f | %15.7f ",
                metrics["epoch"],
                metrics.get("train", np.nan),
                metrics["valid"],
            )

    def configure_optimizers(self):
        """Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for
        training.

        Returns
        -------
        torch.optim.Adam
            The intialized Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), **self.opt_kwargs)


class ZeroWeightedMSELoss:
    def __init__(self, zero_weight=1):
        """Downweight zeros."""
        self.zero_weight = zero_weight

    def __call__(self, input, target):
        """Do the thing"""
        loss = (input - target) ** 2
        loss[target == 0] = loss[target == 0] * self.zero_weight
        return torch.mean(loss)
