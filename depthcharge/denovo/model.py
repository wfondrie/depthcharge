"""A de novo peptide sequencing model"""
import time
import logging
from functools import partial

import torch
import numpy as np
import pandas as pd

from ..masses import PeptideMass
from ..components import SpectrumEncoder, PeptideDecoder

LOGGER = logging.getLogger(__name__)


class Spec2Pep(torch.nn.Module):
    """A Transformer model for de novo peptide sequencing.

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
    max_epochs : int, optional
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
    custom_encoder : SpectrumEncoder, optional
        A pretrained encoder to use. The ``dim_model`` of the encoder must
        be the same as that specified by the ``dim_model`` parameter here.
    max_length : int, optional
        The maximum peptide length to decode.
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
        max_epochs=100,
        updates_per_epoch=500,
        train_batch_size=128,
        eval_batch_size=1000,
        num_workers=1,
        patience=20,
        custom_encoder=None,
        max_length=100,
        n_log=10,
        **kwargs,
    ):
        """Initialize a Spec2Pep model"""
        super().__init__()

        # Writable
        self.max_epochs = max_epochs
        self.updates_per_epoch = updates_per_epoch
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.patience = patience
        self.max_length = max_length
        self.n_log = n_log
        self.fitted = False

        # Build the model
        if custom_encoder:
            self.encoder = custom_encoder
        else:
            self.encoder = SpectrumEncoder(
                dim_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=dropout,
            )

        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.softmax = torch.nn.Softmax(2)

        # Things for training
        self._history = []
        self._opt_kwargs = kwargs
        self._dim_model = dim_model

    @property
    def history(self):
        """The training history."""
        return pd.DataFrame(self._history)

    @property
    def n_parameters(self):
        """The number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, spectra, precursors, sequences):
        """The forward pass

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

    def predict(self, ms_data):
        """Generate the best sequence for a spectrum.

        Currently, we use a greedy decoding; that is, only the best amino
        acid is retained at each decoding step.

        Parameters
        ----------
        ms_data : SpectrumDataset or AnnotatedSpectrumDataset
            The dataset containing the tandem mass spectra to sequence.

        Returns
        -------
        sequences : list or str
            The sequence for each spectrum.
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The score for each amino acid.
        """
        loader = torch.utils.data.DataLoader(
            ms_data,
            batch_size=self.eval_batch_size,
            collate_fn=prepare_batch,
        )

        sequences = []
        scores = []
        self.eval()
        with torch.no_grad():
            for spectra, precursors, *_ in loader:
                spectra = spectra.to(self.encoder.device)
                precursors = precursors.to(self.decoder.device)
                b_scores, b_tokens = self.greedy_decode(spectra, precursors)
                scores.append(b_scores)
                sequences += [self.decoder.detokenize(t) for t in b_tokens]

        return sequences, scores

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

    def fit(self, ms_data):
        """Train the model using a AnnotatedSpectrumDataset

        Parameters
        ----------
        ms_data : AnnotatedSpectrumDataset
            The AnnotatedSpectrumDataset contaiing labeled mass spectra to
            use for training.
        """
        loader = torch.utils.data.DataLoader(
            ms_data,
            batch_size=self.train_batch_size,
            collate_fn=prepare_batch,
            pin_memory=True,
        )

        opt = torch.optim.Adam(self.parameters(), **self._opt_kwargs)
        cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=0)

        LOGGER.info("Training the Spec2Pep model:")
        LOGGER.info("--------------------------------------")
        LOGGER.info("  Epoch |     CE Loss     | Time (s)  ")
        LOGGER.info("--------------------------------------")
        epoch = len(self._history)
        self.train()

        # The main training loop
        try:
            n_updates = 0
            start_time = time.time()
            while epoch <= self.max_epochs:
                for spectra, precursors, sequences in loader:
                    if not n_updates:
                        ce_loss = []

                    spectra = spectra.to(self.encoder.device)
                    precursors = precursors.to(self.encoder.device)
                    opt.zero_grad()
                    pred, truth = self(spectra, precursors, sequences)
                    # The prediction is one longer than we want:
                    pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size)
                    truth = truth.flatten()
                    loss = cross_entropy(pred, truth)
                    loss.backward()
                    opt.step()
                    n_updates += 1
                    ce_loss.append(loss.item())

                    if n_updates == self.updates_per_epoch:
                        epoch += 1
                        ce_loss = np.mean(ce_loss)
                        self._history.append({"cross_entropy_loss": ce_loss})
                        n_updates = 0  # reset for next epoch.

                    if not epoch % self.n_log and not n_updates:
                        LOGGER.info(
                            "  %5i | %15.7f | %8.2f",
                            epoch,
                            ce_loss,
                            time.time() - start_time,
                        )
                        start_time == time.time()

            LOGGER.info("--------------------------------------")
            LOGGER.info("Training complete.")

        except KeyboardInterrupt:
            LOGGER.info("--------------------------------------")
            LOGGER.info("Keyboard interrupt. Stopping...")

        self.fitted = True
        return self


def prepare_batch(batch):
    """This is the collate function"""
    spec, mz, charge, seq = list(zip(*batch))
    charge = torch.tensor(charge)
    mass = (torch.tensor(mz) - 1.007276) * charge
    precursors = torch.vstack([mass, charge]).T.float()
    spec = torch.nn.utils.rnn.pad_sequence(spec, batch_first=True)
    return spec, precursors, seq
