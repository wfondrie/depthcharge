"""Transformer models for peptides and small molecules."""

import warnings

import torch

from ..encoders import PositionalEncoder
from ..mixins import ModelMixin, TransformerMixin
from ..tokenizers import Tokenizer
from .layers import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class _AnalyteTransformer(torch.nn.Module, ModelMixin, TransformerMixin):
    """A transformer base class for analyte sequences.

    Parameters
    ----------
    n_tokens : int or Tokenizer
        The number of tokens used to tokenize molecular sequences.
    d_model : int
        The latent dimensionality to represent each element in the molecular
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
    padding_int : int, optional
        The index that represents padding in the input sequence. Required
        only if ``n_tokens`` was provided as an ``int``.

    """

    def __init__(
        self,
        n_tokens: int | Tokenizer,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float,
        positional_encoder: PositionalEncoder | bool,
        padding_int: int | None,
    ) -> None:
        """Initialize an AnalyteTransformer."""
        super().__init__()
        self._d_model = d_model
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward
        self._n_layers = n_layers
        self._dropout = dropout

        try:
            self._n_tokens = len(n_tokens)
            self._padding_int = n_tokens.padding_int
        except TypeError:
            self._n_tokens = n_tokens
            self._padding_int = padding_int

        if padding_int is not None and padding_int != self._padding_int:
            warnings.warn(
                "The provided padding_int differs from the "
                "Tokenizer.padding_int. The padding_int is being overridden."
            )
        elif padding_int is None and self._padding_int is None:
            raise ValueError(
                "padding_int must be specified when n_tokens is an int.",
            )

        if callable(positional_encoder):
            self.positional_encoder = positional_encoder
        elif positional_encoder:
            self.positional_encoder = PositionalEncoder(d_model)
        else:
            self.positional_encoder = torch.nn.Identity()

        self.token_encoder = torch.nn.Embedding(
            self._n_tokens + 1,
            d_model,
            padding_idx=self._padding_int,
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
        ion mobility of an analyte.

        The representation returned by this method is preprended to the
        peak representations that are fed into the Transformer and
        ultimately contribute to the analyte representation that is the
        first element of the sequence in the model output.

        By default, this method returns a tensor of zeros.

        Parameters
        ----------
        tokens : list of str, torch.Tensor, or None
            The partial molecular sequences for which to predict the next
            token. Optionally, these may be the token indices instead
            of a string.
        *args : torch.Tensor
            Additional data passed with the batch.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The global token representations.

        """
        return torch.zeros((tokens.shape[0], self.d_model)).type_as(
            self.token_encoder.weight
        )


class AnalyteTransformerEncoder(_AnalyteTransformer):
    """A transformer encoder for peptide and small molecule analytes.

    Parameters
    ----------
    n_tokens : int or Tokenizer
        The number of tokens used to tokenize molecular sequences.
    d_model : int
        The latent dimensionality to represent each element in the molecular
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
    padding_int : int, optional
        The index that represents padding in the input sequence. Required
        only if ``n_tokens`` was provided as an ``int``.
    attention_backend : str, optional
        Attention implementation: "sdpa" (default) or "native".
    rotary_embedding : RotaryEmbedding, optional
        Rotary position embedding module to apply to Q and K in self-attention.
        Only compatible with `attention_backend="sdpa"`.
        If ``None``, no rotary embeddings are used.

    """

    def __init__(
        self,
        n_tokens: int | Tokenizer,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        padding_int: int | None = None,
        attention_backend: str = "sdpa",
        rotary_embedding: torch.nn.Module | None = None,
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
            padding_int=padding_int,
        )

        # The Transformer layers:
        layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
            dropout=self.dropout,
            attention_backend=attention_backend,
            rotary_embedding=rotary_embedding,
            enable_sdpa_math=True,
            enable_sdpa_mem_efficient=True,
            enable_sdpa_flash_attention=True,
        )

        self.transformer_encoder = TransformerEncoder(
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
        tokens : torch.Tensor of size (batch_size, len_sequence)
            The integer tokens describing each analyte sequence, padded
            to the maximum analyte length in the batch with 0s.
        *args : torch.Tensor, optional
            Additional data. These may be used by overwriting the
            `global_token_hook()` method in a subclass.
        mask : torch.Tensor
            Passed to `depthcharge.transformers.TransformerEncoder.forward()`.
            The mask for the sequence.
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
        src_key_padding_mask = ~encoded.sum(dim=2).bool()
        src_key_padding_mask[:, 0] = False

        # Add positional encodings
        encoded = self.positional_encoder(encoded)

        # Run through the model:
        latent = self.transformer_encoder(
            encoded,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        return latent, src_key_padding_mask


class AnalyteTransformerDecoder(_AnalyteTransformer):
    """A transformer decoder for peptide or small molecule sequences.

    Parameters
    ----------
    n_tokens : int or Tokenizer
        The number of tokens used to tokenize molecular sequences.
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
        The positional encodings to use for the molecular sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    padding_int : int, optional
        The index that represents padding in the input sequence. Required
        only if ``n_tokens`` was provided as an ``int``.
    attention_backend : str, optional
        Attention implementation: "sdpa" (default) or "native".
    rotary_embedding : RotaryEmbedding, optional
        Rotary position embedding module to apply to Q and K in self-attention.
        Only compatible with `attention_backend="sdpa"`.
        If ``None``, no rotary embeddings are used.

    """

    def __init__(
        self,
        n_tokens: int | Tokenizer,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        padding_int: int | None = None,
        attention_backend: str = "sdpa",
        rotary_embedding: torch.nn.Module | None = None,
    ) -> None:
        """Initialize a AnalyteDecoder."""
        super().__init__(
            n_tokens=n_tokens,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            positional_encoder=positional_encoder,
            padding_int=padding_int,
        )

        # Additional model components
        layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            attention_backend=attention_backend,
            rotary_embedding=rotary_embedding,
            enable_sdpa_math=True,
            enable_sdpa_mem_efficient=True,
            enable_sdpa_flash_attention=True,
        )

        self.transformer_decoder = TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        self.final = torch.nn.Linear(
            d_model,
            self.token_encoder.num_embeddings - 1,
        )

    def embed(
        self,
        tokens: torch.Tensor | None,
        *args: torch.Tensor,
        memory: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """Embed a collection of sequences.

        Parameters
        ----------
        tokens : list of str, torch.Tensor, or None
            The partial molecular sequences for which to predict the next
            token. Optionally, these may be the token indices instead
            of a string.
        *args : torch.Tensor, optional
            Additional data. These may be used by overwriting the
            `global_token_hook()` method in a subclass.
        memory : torch.Tensor of shape (batch_size, len_seq, d_model)
            The representations from a ``TransformerEncoder``, such as a
            ``SpectrumTransformerEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, len_seq)
            Passed to `depthcharge.transformers.TransformerEncoder.forward()`.
            The mask that indicates which elements of ``memory`` are padding.
        memory_mask : torch.Tensor
            Passed to `depthcharge.transformers.TransformerEncoder.forward()`.
            The mask for the memory sequence.
        tgt_mask : torch.Tensor or None
            Passed to `depthcharge.transformers.TransformerEncoder.forward()`.
            The default is a mask that is suitable for predicting the next
            element in the sequence.
        **kwargs : dict
            Additional data fields. These may be used by overwriting
            the `global_token_hook()` method in a subclass.

        Returns
        -------
        embeddings : torch.Tensor of size (batch_size, len_sequence, d_model)
            The output of the Transformer layer containing the embeddings
            of the tokens in the sequence. These may be tranformed to yield
            scores for token predictions using the `.score_embeddings()`
            method.

        """
        # Prepare sequences
        if tokens is None:
            tokens = torch.tensor([[]]).to(self.device)

        # Encode everything:
        encoded = self.token_encoder(tokens)

        # Add the global token
        global_token = self.global_token_hook(tokens, *args, **kwargs)
        encoded = torch.cat([global_token[:, None, :], encoded], dim=1)

        # Create the padding mask:
        tgt_key_padding_mask = encoded.sum(axis=2) == 0
        tgt_key_padding_mask[:, 0] = False

        # Feed through model:
        encoded = self.positional_encoder(encoded)

        output = self.transformer_decoder(
            tgt=encoded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_mask=memory_mask,
        )

        return output

    def score_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Score the embeddings to find the most confident tokens.

        Parameters
        ----------
        embeddings: torch.Tensor of shape (batch_size, len_seq, d_model)
            The embeddings from the Transformer layer.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_tokens)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each token for the
            prediction.

        """
        return self.final(embeddings)

    def forward(
        self,
        tokens: torch.Tensor | None,
        *args: torch.Tensor,
        memory: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """Decode a collection of sequences.

        Parameters
        ----------
        tokens : list of str, torch.Tensor, or None
            The partial molecular sequences for which to predict the next
            token. Optionally, these may be the token indices instead
            of a string.
        *args : torch.Tensor, optional
            Additional data. These may be used by overwriting the
            `global_token_hook()` method in a subclass.
        memory : torch.Tensor of shape (batch_size, len_seq, d_model)
            The representations from a ``TransformerEncoder``, such as a
            ``SpectrumTransformerEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, len_seq)
            Passed to `depthcharge.transformers.TransformerEncoder.forward()`.
            The mask that indicates which elements of ``memory`` are padding.
        memory_mask : torch.Tensor
            Passed to `depthcharge.transformers.TransformerEncoder.forward()`.
            The mask for the memory sequence.
        tgt_mask : torch.Tensor or None
            Passed to `depthcharge.transformers.TransformerEncoder.forward()`.
            The default is a mask that is suitable for predicting the next
            element in the sequence.
        **kwargs : dict
            Additional data fields. These may be used by overwriting
            the `global_token_hook()` method in a subclass.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_tokens)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each token for the
            prediction.

        """
        emb = self.embed(
            tokens,
            *args,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_mask=memory_mask,
            tgt_mask=tgt_mask,
            **kwargs,
        )
        return self.score_embeddings(emb)
