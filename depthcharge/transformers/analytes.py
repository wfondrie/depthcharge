"""Transformer models for peptides and small molecules."""
import torch

from .. import utils
from ..encoders import PositionalEncoder
from ..mixins import ModelMixin, TransformerMixin
from ..tokenizers import (
    MoleculeTokenizer,
    PeptideIonTokenizer,
    PeptideTokenizer,
)


class _AnalyteTransformer(torch.nn.Module, ModelMixin, TransformerMixin):
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
        n_tokens: int
        | PeptideTokenizer
        | PeptideIonTokenizer
        | MoleculeTokenizer,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float,
        positional_encoder: PositionalEncoder | bool,
    ) -> None:
        """Initialize an AnalyteTransformer."""
        super().__init__()
        self._d_model = d_model
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward
        self._n_layers = n_layers
        self._dropout = dropout

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

        self.token_encoder = torch.nn.Embedding(
            n_tokens + 1,
            d_model,
            padding_idx=0,
        )

    def global_token_hook(
        self,
        tokens: torch.Tensor,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """Define how additional information in the batch may be used.

        Overwrite this method to define custom functionality dependent on
        information in the batch. Examples would be to incorporate any
        combination of the mass, charge, retention time, or
        ion mobility of a peptide.

        The representation returned by this method is preprended to the
        peak representations that are fed into the Transformer and
        ultimately contribute to the peptide representation that is the
        first element of the sequence in the model output.

        By default, this method returns a tensor of zeros.

        Parameters
        ----------
        tokens : list of str, torch.Tensor, or None
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        *args : torch.Tensor
            Additional data passed with the batch.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The precursor representations.
        """
        return torch.zeros((tokens.shape[0], self.d_model)).type_as(
            self.token_encoder.weight
        )


class AnalyteTransformerEncoder(_AnalyteTransformer):
    """A transformer encoder for peptide and small molecule analytes.

    Parameters
    ----------
    n_tokens : int, PeptideTokenizer, or MoleculeTokenizer
        The number of tokens used to tokenize analyte sequences.
    d_model : int
        The latent dimensionality to represent the tokens in the analyte
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
        The positional encodings to use for the elements of the sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    """

    def __init__(
        self,
        n_tokens: int | PeptideTokenizer | MoleculeTokenizer,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
    ) -> None:
        """Initialize an AnalyteEncoder."""
        super().__init__(
            n_tokens=n_tokens,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            positional_encoder=positional_encoder,
        )

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
            dropout=self.dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *args: torch.Tensor,
        mask: torch.Tensor = None,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a collection of sequences.

        Parameters
        ----------
        tokens : torch.Tensor of size (batch_size, peptide_length)
            The integer tokens describing each analyte sequence, padded
            to the maximum analyte length in the batch with 0s.
        *args : torch.Tensor, optional
            Additional data. These may be used by overwriting the
            `global_token_hook()` method in a subclass.
        mask : torch.Tensor
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask
            for the sequence.
        **kwargs : dict
            Additional data fields. These may be used by overwriting
            the `global_token_hook()` method in a subclass.

        Returns
        -------
        latent : torch.Tensor of shape (batch_size, len_sequence, d_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        # Encode everything:
        encoded = self.token_encoder(tokens)
        global_token = self.global_token_hook(tokens, *args, **kwargs)
        encoded = torch.cat([global_token[:, None, :], encoded], dim=1)

        # Create mask
        key_mask = ~encoded.sum(dim=2).bool()

        # Add positional encodings
        encoded = self.positional_encoder(encoded)

        # Run through the model:
        latent = self.transformer_encoder(
            encoded,
            mask=mask,
            src_key_padding_mask=key_mask,
        )
        return latent, key_mask


class AnalyteTransformerDecoder(_AnalyteTransformer):
    """A transformer decoder for peptide or small molecule sequences.

    Parameters
    ----------
    n_tokens : int, PeptideTokenizer, or MoleculeTokenizer
        The number of tokens used to tokenize peptide sequences.
    d_model : int, optional
        The latent dimensionality to represent elements of the sequence.
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
    """

    def __init__(
        self,
        n_tokens: int | PeptideTokenizer | MoleculeTokenizer,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
    ) -> None:
        """Initialize a PeptideDecoder."""
        super().__init__(
            n_tokens=n_tokens,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            positional_encoder=positional_encoder,
        )

        # Additional model components
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
            self.token_encoder.num_embeddings - 1,
        )

    def forward(
        self,
        tokens: torch.Tensor | None,
        *args: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a collection of sequences.

        Parameters
        ----------
        tokens : list of str, torch.Tensor, or None
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        *args : torch.Tensor, optional
            Additional data. These may be used by overwriting the
            `global_token_hook()` method in a subclass.
        memory : torch.Tensor of shape (batch_size, len_seq, d_model)
            The representations from a ``TransformerEncoder``, such as a
            ``SpectrumTransformerEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, len_seq)
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask that
            indicates which elements of ``memory`` are padding.
        memory_mask : torch.Tensor
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask
            for the memory sequence.
        tgt_mask : torch.Tensor or None
            Passed to `torch.nn.TransformerEncoder.forward()`. The default
            is a mask that is suitable for predicting the next element in
            the sequence.
        **kwargs : dict
            Additional data fields. These may be used by overwriting
            the `global_token_hook()` method in a subclass.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_tokens)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each amino acid for the
            prediction.

        """
        # Prepare sequences
        if tokens is None:
            tokens = torch.tensor([[]]).to(self.device)

        # Encode everything:
        encoded = self.token_encoder(tokens)
        global_token = self.global_token_hook(tokens, *args, **kwargs)
        encoded = torch.cat([global_token[:, None, :], encoded], dim=1)

        # Feed through model:
        tgt_key_padding_mask = encoded.sum(axis=2) == 0
        encoded = self.positional_encoder(encoded)

        if tgt_mask is None:
            tgt_mask = utils.generate_tgt_mask(encoded.shape[1]).to(
                self.device
            )

        preds = self.transformer_decoder(
            tgt=encoded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_mask=memory_mask,
        )
        return self.final(preds)
