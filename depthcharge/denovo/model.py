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
        self._peptide_calc = PeptideMass(extended=True)

        self._idx2aa = {
            i: aa for i, aa in enumerate(self._peptide_calc.masses.keys())
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
        vocab_size = len(self._peptide_calc) + 1
        self.embedder = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, self._d_model),
            PosEncoder(self._d_model)
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
        self.final = torch.nn.Linear(self._d_model, vocab_size)

        # The optimizer and loss
        self.celoss = torch.nn.CrossEntropyLoss()
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

    def forward(self, spec, mass, seq):
        """The forward pass"""
        # Encode:
        spec, mem_mask = self.spectrum_transformer(spec.to(self._device))

        # Decode:
        preds = [self._force_feed(*d) for d in zip(spec, seq, mass, mem_mask)]
        preds, truth = list(zip(*preds))
        return torch.cat(preds, dim=0), torch.cat(truth, dim=0)

    def _force_feed(self, memory, tgt, mass, mem_mask=None):
        """Force feed the network a peptide sequecnce"""
        # Prepare sequence:
        tgt = tgt.replace("I", "L")
        seq = list(reversed(re.split(r"(?<=.)(?=[A-Z])", tgt)))
        tokens = [self._aa2idx[aa] for aa in seq + ["<eop>"]]
        tokens = torch.tensor(tokens, device=self._device)[None, :]

        # Calculate the remaining mass at each step:
        # Dims should be position (n) x batch (b) x feature (f) for input into
        # the transformer.
        tags = [0] + [self._peptide_calc.mass(seq[:i]) for i in range(len(seq))]
        tags += tags[-1:]
        tags = mass - torch.tensor(tags).to(self._device)[None, :, None]
        tags = self.massdiff_embedder(tags)

        # The input embedding:
        Y = self.embedder(tokens)
        Y = einops.repeat(Y, "b n f -> n (m b) f", m=tags.shape[1])
        Y = torch.cat([tags, Y], dim=0)
        X = einops.repeat(memory, "t f -> t b f", b=Y.shape[1])
        mem_mask = einops.repeat(mem_mask, "t -> n t", n=Y.shape[0])
        mask = generate_mask(Y.shape[1])
        preds = self.decoder(
            tgt=Y,
            memory=X,
            tgt_mask=mask,
            memory_key_padding_mask=mem_mask,
        )
        preds = self.final(einops.rearrange(preds, "n b f -> b n f"))
        return torch.diagonal(preds).T[:-1, :], tokens.squeeze()

    def predict(self, spec, mass):
        """Get the best prediction for a spectrum"""
        spec, mem_mask = self.spectrum_transformer(spec.to(self._device))

        tokens = []
        seqs = []
        scores = []
        for idx in range(spec.shape[0]):
            token, score, seq = self._greedy_decode(
                spec[[idx], :, :], mass[idx].item(), mem_mask[[idx], :]
            )
            tokens.append(token)
            scores.append(score)
            seqs.append(seq)

        return tokens, scores, seqs

    def _greedy_decode(self, spec, mass, mem_mask):
        """Greedy decode each spectrum"""
        #spec = einops.rearrange(spec, "n b f -> b n f")
        Y = torch.tensor([mass], device=self._device)[None, None, :]
        Y = self.massdiff_embedder(Y)
        pred = self.decoder(
            tgt=Y, memory=spec, memory_key_padding_mask=mem_mask
        )
        pred = self.final(einops.rearrange(pred, "n b f -> b n f"))
        tokens = [torch.argmax(pred, dim=2)]
        scores = [pred[[0], -1, :]]

        # Iterate, keeping only the best scoring sequence.
        for _ in range(100):
            seq = [self._idx2aa[i.item()] for i in tokens]
            if seq[-1] == "<eop>":
                break

            mdiff = mass - self._peptide_calc.mass(seq)
            mdiff = torch.tensor([mdiff], device=self._device)[None, None, :]
            mdiff = self.massdiff_embedder(mdiff)

            Y = self.embedder(torch.cat(tokens, dim=0))
            Y = torch.cat([mdiff, Y], dim=0)
            Y = einops.rearrange(Y, "b n f -> n b f")
            pred = self.decoder(
                tgt=Y, memory=spec, memory_key_padding_mask=mem_mask
            )
            pred = self.final(einops.rearrange(pred, "n b f -> b n f"))
            tokens.append(torch.argmax(pred, dim=2)[[-1], :])
            scores.append(pred[[0], -1, :])

        tokens = torch.cat(tokens, dim=0).squeeze()
        return tokens, torch.cat(scores, dim=0), "".join(seq)


    def fit(self, train_dataset, val_dataset=None, encoder_dataset=None):
        """Train the model using a AnnotatedSpectrumDataset

        If a validation set is provided, early stopping can also be
        enabled.

        Parameters
        ----------
        train_dataset : AnnotatedSpectrumDataset
            The AnnotatedSpectrumDataset contaiing labeled mass spectra to
            use for training.
        val_dataset : AnnotatedSpectrumDataset, optional
            The AnnoatedSpectrumDAtaset containing labeld mass spectra to
            use for validation.
        encoder_dataset : SpectrumDataset, optional
            The SpectrumDataset to use for training the encoder.
        """
        self.to(self._device)
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

        # Evaluate the untrained model
        self._start = time.time()
        self._checkpoint()

        # The main training loop
        try:
            self.optimizer.zero_grad()
            for self.epoch in range(self.epoch, self.n_epochs):
                self._start = time.time()
                for specs, seqs, mass in tqdm(self._train_loader, leave=False):
                    preds, truth = self(specs, seqs, mass)
                    loss = self.celoss(preds, truth)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self._checkpoint()

            self.save(self.output_dir / (self.file_root + "final.pt"))

        except KeyboardInterrupt:
            pass

        self.optimizer.zero_grad()
        del specs, seqs
        self.to("cpu")
        self.save(self.output_dir / (self.file_root + "latest.pt"))

    def _checkpoint(self):
        """Evaluate performance and save the best models."""
        with torch.no_grad():
            self.eval()
            self._train_loss.append(self._calc_loss(self._eval_loader[0]))

            if self._val_dataset is not None:
                tail_char = ""
                self._val_loss.append(self._calc_loss(self._eval_loader[1]))
                if self._val_loss[-1] <= self._val_min:
                    self._val_min = self._val_loss[-1]
                    tail_char = "*"
                    self.save(
                        self.output_dir / (self.file_root + "checkpoint.pt")
                    )

                LOGGER.info(
                    "Epoch %i: Train CE %.3g, Val CE %.3g [%3g min]%s",
                    self.epoch,
                    self._train_loss[-1],
                    self._val_loss[-1],
                    (time.time() - self._start) / 60,
                    tail_char,
                )
            else:
                LOGGER.info(
                    "Epoch %i: Train CE %.3g [%g min]",
                    self.epoch,
                    self._train_loss[-1],
                    (time.time() - self._start) / 60,
                )

            self.save(self.output_dir / (self.file_root + "latest.pt"))
            self.train()

    def _calc_loss(self, loader):
        """Calculate the loss of the entire dataset"""
        res = [self(*s) for s in tqdm(loader, leave=False)]
        pred, truth = list(zip(*res))
        pred = torch.cat(pred)
        truth = torch.cat(truth)
        return self.celoss(pred, truth).item()

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
    """The positional encoder for m/z values

    Parameters
    ----------
    d_model : int
        The number of features to output.
    """

    def __init__(self, d_model, max_wavelength=10000):
        """Initialize the MzEncoder"""
        super().__init__()

        n_sin = int(d_model / 2)
        n_cos = d_model - n_sin
        scale = max_wavelength / (2 * np.pi)

        sin_term = scale ** (torch.arange(0, n_sin).float() / (n_sin-1))
        cos_term = scale ** (torch.arange(0, n_cos).float() / (n_cos-1))
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
        pos = torch.arange(X.shape[0])
        pos = einops.repeat(pos, "n -> n b", b=X.shape[1])
        sin_in = einops.repeat(pos, "n b -> n b f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "n b -> n b f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], dim=2)
        return encoded + X


def prepare_batch(batch):
    """This is the collate function"""
    spec, prec, seq = list(zip(*batch))
    mz, charge = list(zip(*prec))
    mass = (torch.tensor(mz) - 1.007276) * torch.tensor(charge)
    spec = torch.nn.utils.rnn.pad_sequence(spec, batch_first=True)
    return spec, mass, seq


def generate_mask(sz):
        """Generate a square mask for the sequence. The masked positions
        are filled with float('-inf'). Unmasked positions are filled with
        float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float('-inf'))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
