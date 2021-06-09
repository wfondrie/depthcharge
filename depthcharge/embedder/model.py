"""The depthcharge embedding model"""
import time
import logging
from pathlib import Path
from functools import partial

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class SpectrumTransformer(torch.nn.Module):
    """A Transformer model for spectrum embedding

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
        embed_dim,
        d_model=128,
        n_head=8,
        d_transformer_fc=1024,
        n_transformer_layers=1,
        n_mlp_layers=3,
        dropout=0,
        norm_reg=0,
        mz_encoding=True,
        n_epochs=300,
        updates_per_epoch=500,
        train_batch_size=1028,
        eval_batch_size=2000,
        num_workers=1,
        patience=10,
        use_gpu=True,
        output_dir=None,
        file_root=None,
        **kwargs,
    ):
        """Initialize a SpectrumTransformer"""
        super().__init__()

        # Readable:
        self._embed_dim = int(embed_dim)
        self._d_model = int(d_model)
        self._n_head = int(n_head)
        self._d_transformer_fc = int(d_transformer_fc)
        self._n_transformer_layers = int(n_transformer_layers)
        self._n_mlp_layers = int(n_mlp_layers)
        self._dropout = float(dropout)
        self._mz_encoding = bool(mz_encoding)
        self._use_gpu = bool(use_gpu)

        # Writable:
        self.n_epochs = int(n_epochs)
        self.updates_per_epoch = int(updates_per_epoch)
        self.norm_reg = float(norm_reg)
        self.train_batch_size = int(train_batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.num_workers = int(num_workers)
        self.patience = int(patience)
        self.use_gpu = bool(use_gpu)
        self.file_root = file_root
        self.output_dir = output_dir

        # Check divisibility
        if self._d_model % self._n_head != 0:
            raise ValueError("n_heads must divide evenly into embed_dim")

        # BUILD THE MODEL
        # First the mz encoding:
        if self._mz_encoding:
            self.mz_encoder = MzEncoder(self._d_model)
        else:
            self.mz_encoder = torch.nn.Linear(2, self._d_model)

        # We need a token to represent the spectrum:
        self.spectrum_token = torch.nn.Parameter(
            torch.randn(1, 1, self._d_model)
        )

        # The transformer layers:
        transformer_encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=self._d_model,
            nhead=self._n_head,
            dim_feedforward=self._d_transformer_fc,
            dropout=self._dropout,
        )

        transformer_encoder = torch.nn.TransformerEncoder(
            transformer_encoder_layers, num_layers=self._n_transformer_layers
        )

        self.transformer = TransformerAdapter(transformer_encoder)

        # The MLP layers:
        # Linearly interpolate the FC layer sizes
        fc_sizes = np.linspace(
            self._d_model, self._embed_dim, self._n_mlp_layers + 1
        )
        fc_sizes = np.ceil(fc_sizes).astype(int)
        fc_layers = []
        for idx in range(len(fc_sizes) - 1):
            fc_layers.append(torch.nn.Linear(fc_sizes[idx], fc_sizes[idx + 1]))
            if idx < len(fc_sizes) - 2:
                fc_layers += [
                    torch.nn.Dropout(p=self._dropout),
                    torch.nn.ReLU(),
                ]
            else:
                fc_layers.append(torch.nn.Softplus())

        if fc_layers:
            self.head = torch.nn.Sequential(*fc_layers)
        else:
            self.head = torch.nn.Softplus()

        # The optimizer and loss:
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), **kwargs)
        self.epoch = 0
        self._base_loader = partial(
            torch.utils.data.DataLoader,
            pin_memory=True,
            collate_fn=prepare_batch,
        )

        # Attributes that are set during training:
        self._train_dataset = None
        self._val_dataset = None
        self._train_loader = None
        self._eval_loader = None
        self._train_loss = None
        self._val_loss = None
        self._val_min = None
        self._start = None

        # For multi-GPU support:
        self.mz_encoder = torch.nn.DataParallel(self.mz_encoder)
        self.transformer = torch.nn.DataParallel(self.transformer)
        self.head = torch.nn.DataParallel(self.head)

    @property
    def loss(self):
        """The loss as a :py:class:`pandas.DataFrame`"""
        n_updates = np.arange(self.epoch) * self.updates_per_epoch
        loss_df = pd.DataFrame(
            {"n_updates": n_updates, "train_mse": self._train_loss}
        )
        if self._val_loss is not None:
            loss_df["validation_mse"] = self._val_loss

        return loss_df

    @property
    def n_parameters(self):
        """The number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def output_dir(self):
        """The directory where models will be saved."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, path):
        """Set the output directory"""
        if path is None:
            path = Path.cwd()

        self._output_dir = Path(path)

    @property
    def file_root(self):
        """The prefix added to the saved files"""
        return self._file_root

    @file_root.setter
    def file_root(self, root):
        """Set the file root"""
        if root is None:
            self._file_root = ""
        elif str(root).endswith("."):
            self._file_root = str(root)
        else:
            self._file_root = str(root) + "."

    @property
    def use_gpu(self):
        """Use one or more GPUs if available."""
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, val):
        """Set whether to use a GPU."""
        self._use_gpu = bool(val)
        if val and torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

    def forward(self, X):
        """The forward pass"""
        mask = torch.cat(
            [
                torch.tensor([[False]] * X.shape[0], device=self._device),
                ~X.sum(dim=2).bool(),
            ],
            dim=1,
        )

        X = self.mz_encoder(X)
        spec = self.spectrum_token.expand(X.shape[0], -1, -1)
        X = torch.cat([spec, X], dim=1)
        return self.transformer(X, mask), mask

    def fit(self, train_dataset, val_dataset=None):
        """Train the model using a SpectrumDataset

        If a validation dataset is provided, early stopping can also
        be enabled.

        Parameters
        ----------
        train_dataset : SpectrumDataset
            The SpectrumDataset containing the training mass spectra.
        val_dataset : SpectrumDataset, optional
            The SpectrumDataset containing the validation mass spectra.

        """
        self.to(self._device)

        # A weird, but necessary way to ensure the optimizer tensors are
        # on the correct device. See this issue for details:
        # https://github.com/pytorch/pytorch/issues/8741#issuecomment-496907204
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        self._train_dataset = train_dataset
        self._train_loader = self._base_loader(
            train_dataset, batch_size=self.train_batch_size
        )

        self._eval_loader = [
            self._base_loader(train_dataset, batch_size=self.eval_batch_size)
        ]

        self._val_dataset = val_dataset
        if self._val_dataset is not None:
            self._eval_loader.append(
                self._base_loader(val_dataset, batch_size=self.eval_batch_size)
            )

        if self._train_loss is None:
            self._train_loss = []

        if self._val_loss is None and self._val_dataset is not None:
            self._val_loss = []
            self._val_min = np.inf

        # Evaluate the untrained model:
        self._start = time.time()
        self._checkpoint()
        self.epoch += 1

        # The main trianing loop:
        try:
            n_updates = 0
            pbar = tqdm(total=self.updates_per_epoch, leave=False)
            for spx, spy, gsp in self._train_loader:
                gsp = gsp.to(self._device)
                emx = self(spx.to(self._device))
                emy = self(spy.to(self._device))
                pred = torch.bmm(emx[:, None, :], emy[:, :, None]).squeeze()
                # pred = self.predict(spx, spy)
                reg = self.norm_reg * (regloss(emx) + regloss(emy))
                loss = self.mse(pred, gsp) + reg
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                pbar.update()
                n_updates += 1

                if n_updates == self.updates_per_epoch:
                    pbar.close()
                    self.epoch += 1
                    self._checkpoint()
                    if self.epoch >= self.n_epochs:
                        break

                    self._start = time.time()
                    n_updates = 0
                    pbar = tqdm(total=self.updates_per_epoch, leave=False)

            self.save(self.output_dir / (self.file_root + "final.pt"))

        except KeyboardInterrupt:
            pbar.close()

        self.optimizer.zero_grad()
        del spx, spy, gsp
        self.to("cpu")
        self.save(self.output_dir / (self.file_root + "latest.pt"))

        return self

    def _checkpoint(self):
        """Evaluate performance and save the best model."""
        with torch.no_grad():
            self.eval()
            self._train_dataset.eval()
            self._train_loss.append(self._calc_loss(self._eval_loader[0]))

            if self._val_dataset is not None:
                tail_char = ""
                self._val_dataset.eval()
                self._val_loss.append(self._calc_loss(self._eval_loader[1]))
                if self._val_loss[-1] <= self._val_min:
                    self._val_min = self._val_loss[-1]
                    tail_char = "*"
                    self.save(
                        self.output_dir / (self.file_root + "checkpoint.pt")
                    )

                LOGGER.info(
                    "Epoch %i: Train MSE %.3g, Val MSE %.3g [%3g min]%s",
                    self.epoch,
                    self._train_loss[-1],
                    self._val_loss[-1],
                    (time.time() - self._start) / 60,
                    tail_char,
                )
            else:
                LOGGER.info(
                    "Epoch %i: Train MSE %.3g [%3g min]",
                    self.epoch,
                    self._train_loss[-1],
                    (time.time() - self._start) / 60,
                )

            self.save(self.output_dir / (self.file_root + "latest.pt"))
            self._train_dataset.train()
            self.train()

    def save(self, path):
        """Save the model weights and metadata

        Parameters
        ----------
        path : str or Path
            The file in which to save the weights and metadata.
        """
        data = {
            "epoch": self.epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": self._train_loss,
            "val_loss": self._val_loss,
        }
        torch.save(data, path)

    def load(self, path):
        """Load the model weights and metadata

        Parameters
        ----------
        path : str or Path
            The file from which to load the weights and metadata.
        """
        data = torch.load(path, map_location=self._device)
        self.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.epoch = data["epoch"]
        self._train_loss = data["train_loss"]
        self._val_loss = data["val_loss"]
        self._val_min = min(data["val_loss"])

    def _calc_loss(self, loader):
        """Calculate the loss over a dataset

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The evaluation data loader.
        """
        pred = []
        gsp = []
        for spx, spy, batch_gsp in tqdm(loader, leave=False):
            gsp.append(batch_gsp)
            pred.append(self.predict(spx, spy))

        pred = torch.cat(pred)
        gsp = torch.cat(gsp)
        return self.mse(pred, gsp.to(self._device)).item()

    def transform(self, X):
        """Embed a mass spectrum

        Parameters
        ----------
        X : torch.Tensor
            The mass spectrum to embed.
        """
        return self(X.to(self._device))[0][:, 0, :]

    def predict(self, X, Y):
        """Embed two batches of spectra and calculate their scalar product.

        Parameters
        ----------
        X : torch.Tensor
        Y : torch.Tensor
            The tensors of spectra to embed and compare.

        Returns
        -------
        torch.Tensor
            The predicted GSP.
        """
        X = self.transform(X)
        X = self.head(X)
        Y = self.transform(Y)
        Y = self.head(Y)
        return torch.bmm(X.unsqueeze(1), Y.unsqueeze(2)).squeeze()


class TransformerAdapter(torch.nn.Module):
    """An adapter so that the transformer works with DataParallel"""

    def __init__(self, transformer):
        """Initialize the TransformerAdapter"""
        super().__init__()
        self.transformer = transformer

    def forward(self, src, src_key_padding_mask):
        """The forward pass"""
        ret = self.transformer(
            src.permute(1, 0, 2), None, src_key_padding_mask
        )
        return ret.permute(1, 0, 2)


class MzEncoder(torch.nn.Module):
    """The positional encoder for m/z values

    Parameters
    ----------
    d_model : int
        The number of features to output.
    """

    def __init__(self, d_model, min_wavelength=0.001, max_wavelength=10000):
        """Initialize the MzEncoder"""
        super().__init__()

        n_sin = int(d_model / 2)
        n_cos = d_model - n_sin
        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength

        sin_term = base * scale ** (torch.arange(0, n_sin).float() / (n_sin-1))
        cos_term = base * scale ** (torch.arange(0, n_cos).float() / (n_cos-1))

        self.linear = torch.nn.Linear(1, d_model, bias=False)
        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor
            A batch of mass spectra.

        Returns
        -------
        torch.Tensor
            The encoded features for the mass spectra.
        """
        mz = X[:, :, [0]]
        sin_mz = torch.sin(mz / self.sin_term)
        cos_mz = torch.cos(mz / self.cos_term)
        encoded = torch.cat([sin_mz, cos_mz], dim=2)
        intensity = self.linear(X[:, :, [1]])
        return encoded + intensity


def prepare_batch(batch):
    """This is the collate function"""
    X, Y, gsp = list(zip(*batch))
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True)
    gsp = torch.tensor(gsp)
    return X, Y, gsp


def regloss(spec):
    return ((1 - torch.linalg.norm(spec, axis=1)) ** 2).mean()
