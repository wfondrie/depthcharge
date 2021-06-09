"""A de novo peptide sequencing model"""
import re
import time
import logging
from pathlib import Path
from functools import partial

import torch
import einops
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..masses import PeptideMass

LOGGER = logging.getLogger(__name__)


class Spec2Pep(torch.nn.Module):
    """A Transformer model for de novo peptide sequencing.

    Parameters
    ----------
    spectrum_transformer : a depthcharge.embedder.SpectrumTransformer object
        The transformer for encoding mass spectra.
    d_transformer_fc : int, optional
        The dimensionality of the fully connected layers within the
        transformer layers of the model. :code:`None` will match the
        `spectrum_transformer`.
    n_transformer_layers : int, optional
        The number of decoder transformer layers. :code:`None` will match the
        `spectrum_transformer`.
    dropout : int, optional
        The dropout rate to use in the decoder layers. :code:`None` will match
        the `spectrum_transformer`
    n_epochs : int, optional
        The maximum number of epochs to train for.
    updates_per_epoch : int, optional
        The number of parameter updates that defines and epoch. :code:`None`
        indicates a full pass through the data.
    train_batch_size : int, optional
        The batch size for training
    eval_batch_size : int, optinoal
        The batch size for evaluation
    num_workers : int, optional
        The number of workers to use for loading batches.
    patience : int, optional
        If the validation set loss has not improved in this many epochs,
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
        spectrum_transformer,
        d_transformer_fc=None,
        n_transformer_layers=None,
        dropout=None,
        n_epochs=100,
        updates_per_epoch=None,
        train_batch_size=128,
        eval_batch_size=1000,
        num_workers=1,
        patience=20,
        use_gpu=True,
        output_dir=None,
        file_root=None,
        residue_dict=None,
        **kwargs,
    ):
        """Initialize a Spec2Pep model"""
        super().__init__()

        # Readable
        self._spectrum_transformer = spectrum_transformer
        self._d_model = spectrum_transformer._d_model
        self._n_head = spectrum_transformer._n_head
        self._n_transformer_layers = spectrum_transformer._n_transformer_layers
        self._d_transformer_fc = spectrum_transformer._d_transformer_fc
        self._dropout = spectrum_transformer._dropout
        self._num_residues = num_residues
        self._peptide_calc = PeptideMass(extended=True)

        self._idx2aa = {
            i: aa for i, aa in enumerate(self.peptide_calc.masses.keys())
        }
        self._idx2aa[len(self._idx2aa)] = "<eop>"
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        if d_transformer_fc is not None:
            self._d_transformer_fc = int(d_transformer_fc)

        if n_transformer_layers is not None:
            self._n_transformer_layers = int(n_transformer_layers)

        if dropout is not None:
            self._dropout = float(dropout)

        # Writable
        self.n_epochs = int(n_epochs)
        self.updates_per_epoch = updates_per_epoch
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = int(num_workers)
        self.patience = int(patience)
        self.use_gpu = use_gpu
        self.file_root = file_root
        self.output_dir = output_dir

        # Build the model
        self.embedder = torch.nn.Sequential(
            torch.nn.embedding(len(residue_dict)), PosEncoder(self._d_model)
        )

        self.massdiff_embedder = torch.nn.Linear(1, self._d_model)

        # The transformer
        transformer_decoder_layers = torch.nn.TransformerDecoderLayer(
            d_model=self._d_model,
            nhead=self._n_head,
            dim_feedforward=self._d_transformer_fc,
            dropout=self._dropout,
        )

        transformer_decoder = torch.nn.TransformerDecoder(
            transformer_decoder_layers, num_layers=self._n_transformer_layers
        )

        self.decoder = TransformerDecoderAdapter(transformer_decoder)
        self.final = torch.nn.Linear(self._d_model, len(residue_dict))

        # The optimizer and loss
        self.celoss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), **kwargs)
        self.epoch = 0
        self._base_loader = partial(
            torch.utils.data.DataLoader,
            pin_memory=True,
            collat_fn=prepare_batch,
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
        self.embedder = torch.nn.DataParallel(self.embedder)
        self.massdiff_embedder = torch.nn.DataParallel(self.massdiff_embedder)
        self.decoder = torch.nn.DataParallel(self.decoder)
        self.final = torch.nn.DataParallel(self.final)

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
    def spectrum_transformer(self):
        """The SpectrumTransformer model used for encoding mass spectra"""
        return self._spectrum_transformer

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

    def forward(self, spec, seq):
        """The forward pass"""
        # Encode:
        spec, mem_mask = self.spectrum_transformer(spec)

        # Decode:
        preds = [self._force_feed(*d) for d in zip(spec, seq, mem_mask)]
        return preds

    def _force_feed(self, memory, tgt, mem_mask=None):
        """Force feed the network a peptide sequecnce"""
        # Prepare sequence:
        seq = list(reversed(re.split(r"(?<=.)(?=[A-Z])", tgt)))
        mass = self._pep_calc(seq)
        tokens = [self._aa2idx[aa] for aa in seq + ["<eop>"]]
        tokens = torch.tensor(tokens, device=self._device)

        # Calculate the remaining mass at each step:
        tags = [0] + [self._pep_calc(seq[:i]) for i in range(len(seq))]
        tags = mass - torch.tensor(tags).to(self._device)
        tags = self.massdiff_embedder(tags)

        # The input embedding:
        Y = self.embed(tokens)
        Y = einops.repeat(Y, "n f -> n b f", b=Y.shape[0])
        Y = torch.cat([tags, Y], dim=0)
        X = einops.repeat(memory, "n f -> n b f", b=Y.shape[0])
        mask = self.decoder.generate_square_subsequent_mask(Y.shape[0])
        preds = self.decoder(
            tgt=Y,
            memory=X,
            tgt_mask=mask,
            memory_key_padding_mask=mem_mask,
        )
        preds = self.fc(einops.rearrange(preds, "t n e -> n t e"))
        return torch.diagonal(preds)


class TransformerDecoderAdapter(torch.nn.Module):
    """An adapter so that the transformer works with DataParallel"""

    def __init__(self, transformer):
        """Initialize the TransformerAdapter"""
        super().__init__()
        self.transformer = transformer

    def forward(self, tgt, memory, **kwargs):
        """The forward pass"""
        ret = self.transformer(
            tgt.permute(1, 0, 2), memory.permute(1, 0, 2), **kwargs
        )
        return ret.permute(1, 0, 2)


class PosEncoder(torch.nn.Module):
    """The positional encoder for amino acids

    Parameters
    ----------
    d_model : int
        The number of features to output.
    """

    def __init__(self, d_model):
        """Initialize the MzEncoder"""
        super().__init__()

        d_model -= 1
        n_sin = int(d_model / 2)
        n_cos = d_model - n_sin

        sin_term = 10000 * torch.exp(
            2 * torch.arange(0, n_sin).float() * (-np.log(3e7) / d_model)
        )
        cos_term = 10000 * torch.exp(
            2 * torch.arange(0, n_cos).float() * (-np.log(3e7) / d_model)
        )

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
        sin_mz = torch.sin(mz * self.sin_term)
        cos_mz = torch.cos(mz * self.cos_term)
        encoded = torch.cat([sin_mz, cos_mz], dim=2)
        raise "This is wrong!"
        # return encoded + X[:, :, [1]]
        return torch.cat([encoded, X[:, :, [1]]], dim=2)


def prepare_batch(batch):
    """This is the collate function"""
    spec, seq = list(zip(*batch))
    spec = torch.nn.utils.rnn.pad_sequence(spec, batch_first=True)
    return spec, seq
