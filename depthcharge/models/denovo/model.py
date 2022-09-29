"""A de novo peptide sequencing model"""
import logging

import torch
import einops
import numpy as np
import pytorch_lightning as pl

from ...components import SpectrumEncoder, PeptideDecoder, ModelMixin

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
    beam_size : int, optional
        The number of paths to pursue when decoding.
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
        beam_size=3,
        n_log=10,
        **kwargs,
    ):
        """Initialize a Spec2Pep model"""
        super().__init__()

        # Writable
        self.max_length = max_length
        self.n_log = n_log
        self.beam_size = beam_size

        # Build the model
        if custom_encoder is not None:
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

        self.softmax = torch.nn.Softmax(dim=2)
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
        sequences : list of list of str
            The top sequences for each spectrum.
        scores : torch.Tensor of shape
        (n_spectra, length, n_amino_acids, beam_size)
            The score for each amino acid.
        """
        spectra = spectra.to(self.encoder.device)
        precursors = precursors.to(self.decoder.device)
        scores, tokens = self.beam_search_decode(spectra, precursors)
        sequences = []
        for spec_tokens in tokens:
            sequences.append(
                [self.decoder.detokenize(t) for t in spec_tokens.T]
            )

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

    def beam_search_decode(self, spectra, precursors):
        """Beam search decode the spectra.

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
        tokens : torch.Tensor of shape
        (n_spectra, max_length, n_amino_acids, beam_size)
            The token sequence for each spectrum.
        scores : torch.Tensor of shape
        (n_spectra, max_length, n_amino_acids, beam_size)
            The score for each amino acid.
        """
        memories, mem_masks = self.encoder(spectra)

        # Sizes:
        batch = spectra.shape[0]  # B
        length = self.max_length + 1  # L
        vocab = self.decoder.vocab_size + 1  # V
        beam = self.beam_size  # S

        # initialize scores and tokens:
        scores = torch.zeros(batch, length, vocab, beam)
        scores = scores.type_as(spectra)
        scores[scores == 0] = torch.nan
        tokens = torch.zeros(batch, length, beam)
        tokens = tokens.type_as(spectra).long()

        # Create the first prediction:
        pred, _ = self.decoder(None, precursors, memories, mem_masks)
        _, idx = torch.topk(pred[:, 0, :], beam, dim=1)
        tokens[:, 0, :] = idx
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Make precursors and memories the right shape for decoding:
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)

        # The main decoding loop:
        for idx in range(1, self.max_length + 1):
            scores = einops.rearrange(scores, "B L V S -> (B S) L V")
            tokens = einops.rearrange(tokens, "B L S -> (B S) L")
            decoded = (tokens == self.stop_token).any(axis=1)
            if decoded.all():
                scores = einops.rearrange(
                    scores,
                    "(B S) L V -> B L V S",
                    S=beam,
                )
                tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
                break

            # Get scores:
            scores[~decoded, : idx + 1, :], _ = self.decoder(
                tokens[~decoded, :idx],
                precursors[~decoded, :],
                memories[~decoded, :, :],
                mem_masks[~decoded, :],
            )

            # Reshape to group by spectrum (B for "batch")
            scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)
            tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
            prev_tokens = einops.repeat(
                tokens[:, :idx, :],
                "B L S -> B L V S",
                V=vocab,
            )

            # Get the previous tokens and scores:
            prev_scores = torch.gather(
                scores[:, :idx, :, :],
                dim=2,
                index=prev_tokens,
            )
            prev_scores = einops.repeat(
                prev_scores[:, :, 0, :],
                "B L S -> B L (V S)",
                V=vocab,
            )

            # Figure out the top K decodings:
            step_scores = torch.zeros(batch, idx + 1, beam * vocab)
            step_scores = step_scores.type_as(scores)
            step_scores[:, :idx, :] = prev_scores
            step_scores[:, idx, :] = einops.rearrange(
                scores[:, idx, :, :],
                "B V S -> B (V S)",
            )

            _, top_idx = torch.topk(step_scores.nanmean(dim=1), beam)
            V_idx, S_idx = np.unravel_index(top_idx, (vocab, beam))

            S_idx = einops.rearrange(S_idx, "B S -> (B S)")
            # V_idx = einops.rearrange(V_idx, "B S -> (B S)")
            B_idx = einops.repeat(torch.arange(batch), "B -> (B S)", S=beam)

            # Record the top K decodings
            tokens[:, :idx, :] = einops.rearrange(
                prev_tokens[B_idx, :, 0, S_idx],
                "(B S) L -> B L S",
                S=beam,
            )
            tokens[:, idx, :] = torch.tensor(V_idx)
            scores[:, : idx + 1, :, :] = einops.rearrange(
                scores[B_idx, : idx + 1, :, S_idx],
                "(B S) L V -> B L V S",
                S=beam,
            )

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
        pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
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
        pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
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
