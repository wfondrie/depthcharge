"""The depthcharge embedding model"""
import logging

import torch
import pytorch_lightning as pl

from ..components import SpectrumEncoder, FeedForward, ModelMixin

LOGGER = logging.getLogger(__name__)


class SiameseSpectrumEncoder(pl.LightningModule, ModelMixin):
    """A Transformer model for self-supervised learning of spectrum embeddings.

    Parameters
    ----------
    embed_dim : int
        The embedding dimension
    d_model : int, optional
        The latent dimension for the spectum and each peak learned by the
        transformer.
    n_head : int, optional
        The number of attention heads in each layer. `d_model` must be divisble
        by `n_head`.
    d_transformer_fc : int, optional
        The dimensionality of the fully connected layers in the
        transformer layers of the model.
    n_transformer_layers : int, optional
        The number of transformer layers.
    n_mlp_layers : int, optional
        The number of hidden layers for the multilayer perceptron head.
    dropout : float, optional
        The dropout rate in all layers of the model.
    mz_encoding : bool, optional
        Use positial encodings for the m/z values of each peak.
    n_epochs : int, optional
        The maximum number of epochs to train for.
    updates_per_epoch : int, optional
        The number of parameter updates that defines an epoch.
    train_batch_size : int, optional
        The batch size for training.
    eval_batch_size : int, optional
        The batch size for evaluation.
    num_workers : int, optional
        The number of workers to use for loading batches of spectra.
    patience : int, optional
        If the validation set loss has not impoved in this many epochs,
        training will stop. :code:`None` disables early stopping.
    use_gpu : str, optional
        Use one or more GPUs if available.
    output_dir : str or Path, optional
        The location to save the training models.
    file_root : str, optional
        A stem to add to the file names for the saved models.
    **kwargs : Dict
        Keyword arguments passed to the Adam optimizer
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        head_layers=3,
        dropout=0,
        max_epochs=300,
        updates_per_epoch=500,
        train_batch_size=1028,
        eval_batch_size=2000,
        num_workers=1,
        patience=20,
        n_log=10,
        **kwargs,
    ):
        """Initialize a SpectrumTransformer"""
        super().__init__()

        # Writable:
        self.max_epochs = max_epochs
        self.updates_per_epoch = updates_per_epoch
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.patience = patience
        self.n_log = n_log
        self.fitted = False

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
            append=torch.nn.Sigmoid(),
        )

        self.mse = torch.nn.MSELoss()
        self.opt_kwargs = kwargs

    def forward(self, X):
        """The forward pass"""
        return self.encoder(X)

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
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True)
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

        LOGGER.info(
            "  %5i | %15.7f | %15.7f ",
            metrics["epoch"],
            metrics["train_loss"],
            metrics["valid_loss"],
        )

    def configure_optimizers(self):
        """Initialize our optimizer"""
        return torch.optim.Adam(self.parameters(), **self.opt_kwargs)
