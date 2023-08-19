"""Transformer models for peptides."""
import torch

from ..encoders import FloatEncoder, PositionalEncoder
from ..tokenizers import PeptideTokenizer


class _PeptideTransformer(torch.nn.Module):
    """A transformer base class for peptide sequences.

    Parameters
    ----------
    n_tokens : int or PeptideTokenizer
        The number of tokens used to tokenize peptide sequences.
    d_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    positional_encoder : PositionalEncoder or bool
        The positional encodings to use for the amino acid sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    max_charge : int
        The maximum precursor charge to embed.
    """

    def __init__(
        self,
        n_tokens: int | PeptideTokenizer,
        d_model: int,
        positional_encoder: PositionalEncoder | bool,
        max_charge: int,
    ) -> None:
        super().__init__()
        try:
            n_tokens = len(n_tokens)
        except TypeError:
            pass

        if callable(positional_encoder):
            self.positional_encoder = positional_encoder
        elif positional_encoder:
            self.positional_encoder = PositionalEncoder(d_model)
        else:
            self.positional_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge + 1, d_model)
        self.aa_encoder = torch.nn.Embedding(
            n_tokens + 1,
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
    n_tokens : int or PeptideTokenizer
        The number of tokens used to tokenize peptide sequences.
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
        n_tokens: int | PeptideTokenizer,
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
            n_tokens=n_tokens,
            d_model=d_model,
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
        tokens: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        tokens : torch.Tensor of size (batch_size, peptide_length)
            The integer tokens describing each peptide sequence, padded
            to the maximum peptide length in the batch with 0s.
        charges : torch.Tensor of size (batch_size,)
            The charge state of each peptide.

        Returns
        -------
        latent : torch.Tensor of shape (n_sequences, len_sequence, d_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        # Encode everything:
        encoded = self.aa_encoder(tokens)
        charges = self.charge_encoder(charges)[:, None]
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
    n_tokens : int or PeptideTokenizer
        The number of tokens used to tokenize peptide sequences.
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
    positional_encoder : PositionalEncoder or bool, optional
        The positional encodings to use for the amino acid sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    max_charge : int, optional
        The maximum charge state for peptide sequences.
    """

    def __init__(
        self,
        n_tokens: int | PeptideTokenizer,
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
            n_tokens=n_tokens,
            d_model=d_model,
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

        self.final = torch.nn.Linear(
            d_model,
            self.aa_encoder.num_embeddings - 1,
        )

    def forward(
        self,
        tokens: torch.Tensor | None,
        precursors: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        tokens : list of str, torch.Tensor, or None
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

        """
        # Prepare sequences
        if tokens is None:
            tokens = torch.tensor([[]]).to(self.device)

        # Encode everything:
        tokens = self.aa_encoder(tokens)
        masses = self.mass_encoder(precursors[:, None, 0])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Feed through model:
        tgt = torch.cat([precursors, tokens], dim=1)
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.positional_encoder(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[1]).to(self.device)
        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        )
        return self.final(preds)


def generate_tgt_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)
