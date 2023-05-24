"""Transformer models for peptides."""
import torch

from .. import utils
from ..encoders import FloatEncoder, PositionalEncoder
from ..tokenizers import PeptideTokenizer


class _PeptideTransformer(torch.nn.Module):
    """A transformer base class for peptide sequences.

    Parameters
    ----------
    d_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    tokenizer : PeptideTokenizer
        A tokenizer specifying how to handle peptide sequences.
    positional_encoder : PositionalEncoder or bool
        The positional encodings to use for the amino acid sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    max_charge : int
        The maximum precursor charge to embed.
    """

    def __init__(
        self,
        d_model: int,
        tokenizer: PeptideTokenizer,
        positional_encoder: PositionalEncoder | bool,
        max_charge: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        if isinstance(positional_encoder, PositionalEncoder):
            self.positional_encoder = positional_encoder
            if positional_encoder.d_model != d_model:
                ValueError("The PositionaEncoder have the same 'd_model'.")
        elif positional_encoder:
            self.positional_encoder = PositionalEncoder(d_model)
        else:
            self.positional_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge, d_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self.tokenizer),
            d_model,
            padding_idx=0,
        )

    @property
    def device(self) -> torch.device:
        """The current device for the model."""
        return next(self.parameters()).device


class PeptideTransformerEncoder(_PeptideTransformer):
    """A transformer encoder for peptide sequences.

    Parameters
    ----------
    tokenizer : PeptideTokenizer
        A tokenizer specifying how to handle peptide sequences.
    d_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    nhead : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``nhead``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    positional_encoder : PositionalEncoder or bool, optional
        The positional encodings to use for the amino acid sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    max_charge : int, optional
        The maximum charge state for peptide sequences.
    """

    def __init__(
        self,
        tokenizer: PeptideTokenizer,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        max_charge: int = 5,
    ) -> None:
        """Initialize a PeptideEncoder."""
        super().__init__(
            d_model=d_model,
            tokenizer=tokenizer,
            positional_encoder=positional_encoder,
            max_charge=max_charge,
        )

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(
        self,
        sequences: list[str] | torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or torch.Tensor of length batch_size
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        charges : torch.Tensor of size (batch_size,)
            The charge state of the peptide

        Returns
        -------
        latent : torch.Tensor of shape (n_sequences, len_sequence, d_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        sequences = utils.listify(sequences)
        if isinstance(sequences[0][0], str):
            tokens = self.tokenizer.tokenize(sequences).to(self.device)
        else:
            tokens = sequences.to(self.device)

        encoded = self.aa_encoder(tokens)

        # Encode charges
        charges = self.charge_encoder(charges - 1)[:, None]
        encoded = torch.cat([charges, encoded], dim=1)

        # Create mask
        mask = ~encoded.sum(dim=2).bool()

        # Add positional encodings
        encoded = self.positional_encoder(encoded)

        # Run through the model:
        latent = self.transformer_encoder(encoded, src_key_padding_mask=mask)
        return latent, mask


class PeptideTransformerDecoder(_PeptideTransformer):
    """A transformer decoder for peptide sequences.

    Parameters
    ----------
    tokenizer : PeptideTokenizer
        A tokenizer specifying how to handle peptide sequences.
    d_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    nhead : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``nhead``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    postional_encoder : PositionalEncoder or bool, optional
        The positional encodings to use for the amino acid sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    max_charge : int, optional
        The maximum charge state for peptide sequences.
    """

    def __init__(
        self,
        tokenizer: PeptideTokenizer,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        max_charge: int = 5,
    ) -> None:
        """Initialize a PeptideDecoder."""
        super().__init__(
            d_model=d_model,
            tokenizer=tokenizer,
            positional_encoder=positional_encoder,
            max_charge=max_charge,
        )

        # Additional model components
        self.mass_encoder = FloatEncoder(d_model)
        layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        self.final = torch.nn.Linear(d_model, len(self.tokenizer))

    def forward(
        self,
        sequences: list[str] | torch.Tensor | None,
        precursors: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str, torch.Tensor, or None
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        precursors : torch.Tensor of size (batch_size, 2)
            The precursor mass (axis 0) and charge (axis 1).
        memory : torch.Tensor of shape (batch_size, n_peaks, d_model)
            The representations from a ``TransformerEncoder``, such as a
           ``SpectrumEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, n_peaks)
            The mask that indicates which elements of ``memory`` are padding.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_amino_acids)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each amino acid for the
            prediction.
        tokens : torch.Tensor of size (batch_size, len_sequence)
            The input padded tokens.

        """
        # Prepare sequences
        if sequences is not None:
            sequences = utils.listify(sequences)
            if isinstance(sequences[0][0], str):
                tokens = self.tokenizer.tokenize(sequences).to(self.device)
            else:
                tokens = sequences.to(self.device)
        else:
            tokens = torch.tensor([[]]).to(self.device)

        # Prepare mass and charge
        masses = self.mass_encoder(precursors[:, None, 0])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Feed through model:
        if sequences is None:
            tgt = precursors
        else:
            tgt = torch.cat([precursors, self.aa_encoder(tokens)], dim=1)

        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.positional_encoder(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[1]).type_as(precursors)
        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        )
        return self.final(preds), tokens


def generate_tgt_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)
