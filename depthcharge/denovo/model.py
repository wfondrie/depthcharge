"""A de novo peptide sequencing model"""
import logging

import torch
import numpy as np
import pytorch_lightning as pl

from ..masses import PeptideMass
from ..components import SpectrumEncoder, PeptideDecoder, ModelMixin
from ..embed.model import SiameseSpectrumEncoder

LOGGER = logging.getLogger(__name__)


class Spec2Pep(pl.LightningModule, ModelMixin):
    """A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality used by the Transformer model.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    dim_intensity : int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value. If ``None``, the intensity will be projected
        up to ``dim_model`` using a linear layer, then summed with the m/z
        emcoding for each peak.
    custom_encoder : SpectrumEncoder or PairedSpectrumEncoder, optional
        A pretrained encoder to use. The ``dim_model`` of the encoder must
        be the same as that specified by the ``dim_model`` parameter here.
    max_length : int, optional
        The maximum peptide length to decode.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int, optional
        The maximum charge state to consider.
    n_log : int, optional
        The number of epochs to wait between logging messages.
    **kwargs : Dict
        Keyword arguments passed to the Adam optimizer
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        dim_intensity=None,
        custom_encoder=None,
        max_length=100,
        residues="canonical",
        max_charge=5,
        n_log=10,
        **kwargs,
    ):
        """Initialize a Spec2Pep model"""
        super().__init__()

        # Writable
        self.max_length = max_length
        self.n_log = n_log

        # Build the model
        if custom_encoder is not None:
            if isinstance(custom_encoder, SiameseSpectrumEncoder):
                self.encoder = custom_encoder.encoder
            else:
                self.encoder = custom_encoder
        else:
            self.encoder = SpectrumEncoder(
                dim_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=dropout,
                dim_intensity=dim_intensity,
            )

        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
        )

        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(ignore_index=0)

        # Things for training
        self._history = []
        self.opt_kwargs = kwargs
        self.stop_token = self.decoder._aa2idx["$"]

    def forward(self, spectra, precursors):
        """Sequence a batch of mass spectra.

        Parameters
        ----------
        spectrum : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        precursors : torch.Tensor of size (n_spectra, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum.

        Returns
        -------
        sequences : list or str
            The sequence for each spectrum.
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The score for each amino acid.
        """
        spectra = spectra.to(self.encoder.device)
        precursors = precursors.to(self.decoder.device)
        scores, tokens = self.greedy_decode2(spectra, precursors)
        sequences = [self.decoder.detokenize(t) for t in tokens]
        return sequences, scores

    def predict_step(self, batch, *args):
        """Sequence a batch of mass spectra.

        Note that this is used within the context of a pytorch-lightning
        Trainer to generate a prediction.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain mass spectra (index 0) and the
            precursor mass and charge (index 1). It may have more indices,
            but these will be ignored.


        Returns
        -------
        sequences : list or str
            The sequence for each spectrum.
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The score for each amino acid.
        """
        return self(batch[0], batch[1])

    def greedy_decode(self, spectra, precursors):
        """Greedy decode the spoectra.

        Parameters
        ----------
        spectrum : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        precursors : torch.Tensor of size (n_spectra, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra, max_length, n_amino_acids)
            The token sequence for each spectrum.
        scores : torch.Tensor of shape (n_spectra, max_length, n_amino_acids)
            The score for each amino acid.
        """
        memories, mem_masks = self.encoder(spectra)
        all_scores = []
        all_tokens = []
        for prec, mem, mask in zip(precursors, memories, mem_masks):
            prec = prec[None, :]
            mem = mem[None, :]
            mask = mask[None, :]
            scores, _ = self.decoder(None, prec, mem, mask)
            tokens = torch.argmax(scores, axis=2)
            for _ in range(self.max_length - 1):
                if self.decoder._idx2aa[tokens[0, -1].item()] == "$":
                    break

                scores, _ = self.decoder(tokens, prec, mem, mask)
                tokens = torch.argmax(scores, axis=2)

            all_scores.append(self.softmax(scores).squeeze())
            all_tokens.append(tokens.squeeze())

        return all_scores, all_tokens

    def greedy_decode2(self, spectra, precursors):
        """Greedy decode the spoectra.

        Parameters
        ----------
        spectrum : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        precursors : torch.Tensor of size (n_spectra, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra, max_length, n_amino_acids)
            The token sequence for each spectrum.
        scores : torch.Tensor of shape (n_spectra, max_length, n_amino_acids)
            The score for each amino acid.
        """
        memories, mem_masks = self.encoder(spectra)

        # initialize scores:
        scores = torch.zeros(
            spectra.shape[0],
            self.max_length + 1,
            self.decoder.vocab_size,
        )
        scores = scores.type_as(spectra)

        # The first predition:
        scores[:, :1, :], _ = self.decoder(
            None,
            precursors,
            memories,
            mem_masks,
        )

        tokens = torch.argmax(scores, axis=2)

        # Keep predicting until all have a stop token or max_length is reached.
        # Don't count the stop token toward max_length though.
        for idx in range(2, self.max_length + 2):
            decoded = (tokens == self.stop_token).any(axis=1)
            if decoded.all():
                break

            scores[~decoded, :idx, :], _ = self.decoder(
                tokens[~decoded, : (idx - 1)],
                precursors[~decoded, :],
                memories[~decoded, :, :],
                mem_masks[~decoded, :],
            )
            tokens = torch.argmax(scores, axis=2)

        return self.softmax(scores), tokens

    def _step(self, spectra, precursors, sequences):
        """The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        precursors : torch.Tensor of size (n_spectra, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum.
        sequences : list or str of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The raw scores for each amino acid at each position.
        tokens : torch.Tensor of shape (n_spectra, length)
            The best token at each sequence position
        """
        memory, mem_mask = self.encoder(spectra)
        scores, tokens = self.decoder(sequences, precursors, memory, mem_mask)
        return scores, tokens

    def training_step(self, batch, *args):
        """A single training step

        Note that this is used within the context of a pytorch-lightning
        Trainer to generate a prediction.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain mass spectra (index 0), the
            precursor mass and charge (index 1), and the peptide sequence
            (index 2)

        Returns
        -------
        torch.Tensor
            The loss.
        """
        spectra, precursors, sequences = batch
        pred, truth = self._step(spectra, precursors, sequences)
        pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size)
        loss = self.celoss(pred, truth.flatten())
        self.log(
            "CELoss",
            {"train": loss.item()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, *args):
        """A single validation step

        Note that this is used within the context of a pytorch-lightning
        Trainer to generate a prediction.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch is expected to contain mass spectra (index 0), the
            precursor mass and charge (index 1), and the peptide sequence
            (index 2)

        Returns
        -------
        torch.Tensor
            The loss.
        """
        spectra, precursors, sequences = batch
        pred, truth = self._step(spectra, precursors, sequences)
        pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size)
        loss = self.celoss(pred, truth.flatten())
        self.log(
            "CELoss",
            {"valid": loss.item()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        """Log the training loss.

        This is a pytorch-lightning hook.
        """
        metrics = {
            "epoch": self.trainer.current_epoch,
            "train": self.trainer.callback_metrics["CELoss"]["train"].item(),
        }
        self._history.append(metrics)

    def on_validation_epoch_end(self):
        """Log the epoch metrics to self.history.

        This is a pytorch-lightning hook.
        """
        if not self._history:
            return

        valid_loss = self.trainer.callback_metrics["CELoss"]["valid"].item()
        self._history[-1]["valid"] = valid_loss

    def on_epoch_end(self):
        """Print log to console, if requested."""
        if len(self._history) == 1:
            LOGGER.info("---------------------------------------")
            LOGGER.info("  Epoch |   Train Loss  |  Valid Loss  ")
            LOGGER.info("---------------------------------------")

        metrics = self._history[-1]
        if not metrics["epoch"] % self.n_log:
            LOGGER.info(
                "  %5i | %13.6f | %13.6f ",
                metrics["epoch"],
                metrics.get("train", np.nan),
                metrics.get("valid", np.nan),
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
