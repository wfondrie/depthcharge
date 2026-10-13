"""Custom Transformer layers for depthcharge."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import combine_key_pad_and_attn, generate_tgt_mask
from .attn import MultiheadAttention


def _validate_attn_mask(
    mask: Tensor,
    tensor: Tensor,
    nhead: int,
    mask_name: str,
    tgt_seq_len: int,
    src_seq_len: int | None = None,
) -> None:
    """Validate attention mask shape and dtype.

    Parameters
    ----------
    mask : Tensor
        The attention mask to validate.
    tensor : Tensor
        The reference tensor for dtype checking.
    nhead : int
        Number of attention heads.
    mask_name : str
        Name of the mask for error messages.
    tgt_seq_len : int
        Target sequence length.
    src_seq_len : int, optional
        Source sequence length. If None, expects square mask (self-attention).
        If provided, expects rectangular mask (cross-attention).

    Raises
    ------
    ValueError
        If mask shape is invalid.
    TypeError
        If mask dtype doesn't match tensor dtype.

    """
    if src_seq_len is None:
        src_seq_len = tgt_seq_len

    batch_size = tensor.size(0)
    valid_shapes = (
        (list(mask.size()) == [tgt_seq_len, src_seq_len])
        or (list(mask.size()) == [batch_size, tgt_seq_len, src_seq_len])
        or (list(mask.size()) == [batch_size, nhead, tgt_seq_len, src_seq_len])
    )

    if not valid_shapes:
        if src_seq_len == tgt_seq_len:
            expected = (
                "(seq_len, seq_len), (batch_size, seq_len, seq_len), or "
                "(batch_size, nhead, seq_len, seq_len)"
            )
        else:
            expected = (
                "(tgt_seq_len, src_seq_len), "
                "(batch_size, tgt_seq_len, src_seq_len), or "
                "(batch_size, nhead, tgt_seq_len, src_seq_len)"
            )
        raise ValueError(f"`{mask_name}` should have size {expected}")

    if mask.dtype != tensor.dtype:
        raise TypeError(
            f"`{mask_name}` should have same dtype as reference tensor"
        )


def _validate_key_padding_mask(
    mask: Tensor,
    batch_size: int,
    seq_len: int,
    mask_name: str,
) -> None:
    """Validate key padding mask shape and dtype.

    Parameters
    ----------
    mask : Tensor
        The key padding mask to validate.
    batch_size : int
        Expected batch size.
    seq_len : int
        Expected sequence length.
    mask_name : str
        Name of the mask for error messages.

    Raises
    ------
    ValueError
        If mask shape is invalid.
    TypeError
        If mask dtype is not bool.

    """
    if list(mask.size()) != [batch_size, seq_len]:
        raise ValueError(
            f"`{mask_name}` should have size (batch_size, seq_len)"
        )

    if mask.dtype != torch.bool:
        raise TypeError(f"`{mask_name}` should be bool")


class TransformerEncoderLayer(nn.Module):
    """Custom Transformer encoder layer.

    Parameters
    ----------
    d_model : int
        The number of expected features in the input.
    nhead : int
        The number of heads in the MultiheadAttention module.
    dim_feedforward : int, optional
        The dimension of the feedforward network model.
    dropout : float, optional
        The dropout value.
    activation : str or callable, optional
        The activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: "relu"
    layer_norm_eps : float, optional
        The eps value in layer normalization components.
    norm_first : bool, optional
        If True, layer norm is done prior to attention and feedforward
        operations (pre-norm). Otherwise it's done after (post-norm).
    batch_first : bool, optional
        Placeholder for API compatibility with
        `torch.nn.TransformerDecoderLayer`.
        Should always be True.
    attention_backend : str, optional
        Attention implementation: "sdpa" (default) or "native".
    rotary_embedding : RotaryEmbedding, optional
        Rotary position embedding module to apply to Q and K.
        If None, no rotary embeddings are used. Default: None
    enable_sdpa_math : bool, optional
        If True, enable SDPA math kernel when using "sdpa" backend.
        Default: True
    enable_sdpa_mem_efficient : bool, optional
        If True, enable SDPA memory efficient kernel when using "sdpa" backend.
        Default: True
    enable_sdpa_flash_attention : bool, optional
        If True, enable SDPA flash attention kernel when using "sdpa" backend.
        Default: True

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | callable = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        batch_first: bool = True,
        attention_backend: str = "sdpa",
        rotary_embedding: nn.Module | None = None,
        enable_sdpa_math: bool = True,
        enable_sdpa_mem_efficient: bool = True,
        enable_sdpa_flash_attention: bool = True,
    ) -> None:
        """Initialize a TransformerEncoderLayer."""
        super().__init__()

        if not batch_first:
            raise ValueError(
                "TransformerEncoderLayer requires batch_first=True"
            )

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.norm_first = norm_first
        self.attention_backend = attention_backend

        if attention_backend == "native":
            if rotary_embedding is not None:
                raise ValueError(
                    "Rotary embeddings are not supported with the "
                    "native attention backend."
                )

            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
        elif attention_backend == "sdpa":
            self.self_attn = MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
                rotary_embedding=rotary_embedding,
                enable_sdpa_math=enable_sdpa_math,
                enable_sdpa_mem_efficient=enable_sdpa_mem_efficient,
                enable_sdpa_flash_attention=enable_sdpa_flash_attention,
            )
        else:
            raise ValueError(
                '`attention_backend` should be one of "sdpa" or "native"'
            )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation = nn.functional.relu
        elif activation == "gelu":
            self.activation = nn.functional.gelu
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError(
                f"activation should be 'relu', 'gelu', or a callable, "
                f"not {activation}"
            )

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
        positions: Tensor | None = None,
    ) -> Tensor:
        """Pass the input through the encoder layer.

        Parameters
        ----------
        src : Tensor
            (batch_size, seq_len, d_model).
        src_mask : Tensor, optional
            If specified, pairwise additive (float) mask
            Shape:
                - (seq_len, seq_len),
                - (batch_size, seq_len, seq_len) or
                - (batch_size, nhead, seq_len, seq_len)
        src_key_padding_mask : Tensor, optional
            If specified, should be a binary mask shape (batch_size, seq_len)
            indicating which elements should participate in attention.
            `True` values indicate positions that should not participate
            in attention (e.g., padded elements).
        is_causal : bool, optional
            If True, applies a causal mask.
        positions : Tensor, optional
            Position values for rotary embeddings.
            If None, uses integer positions.
            Ignored when RotaryEmbedding is not used.
            Shape: (batch, seq_len) or (seq_len,)

        Returns
        -------
        Tensor
            The output of the encoder layer.

        """
        if src_mask is not None:
            _validate_attn_mask(
                src_mask,
                src,
                self.nhead,
                "src_mask",
                src.size(1),
            )
        if src_key_padding_mask is not None:
            _validate_key_padding_mask(
                src_key_padding_mask,
                src.size(0),
                src.size(1),
                "src_key_padding_mask",
            )

        if self.norm_first:
            # Pre-norm architecture
            src = src + self._sa_block(
                self.norm1(src),
                src_mask,
                src_key_padding_mask,
                is_causal,
                positions,
            )
            src = src + self._ff_block(self.norm2(src))
        else:
            # Post-norm architecture (default for PyTorch)
            src = self.norm1(
                src
                + self._sa_block(
                    src,
                    src_mask,
                    src_key_padding_mask,
                    is_causal,
                    positions,
                )
            )
            src = self.norm2(src + self._ff_block(src))

        return src

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
        positions: Tensor | None = None,
    ) -> Tensor:
        """Does self-attention with argument interpretation.

        When causal and attn_mask is given: combine, then don't pass
        is_causal to MultiHeadAttention.
        When causal and no attn_mask: if native MultiHeadAttention,
        then pass is_causal with an explicit mask, else just pass is_causal.

        In addition, combines attn_mask with key_padding_mask if both are
        passed.
        """
        pass_causal = False
        if is_causal:
            if attn_mask is not None:
                causal_mask = generate_tgt_mask(x.shape[-2]).to(x.device)
                attn_mask = attn_mask.masked_fill(
                    causal_mask, -float("inf")
                )  # broadcasts to (s, s), (b, s, s) or (b, nh, s, s)
                pass_causal = False
            else:
                pass_causal = True
                if isinstance(self.self_attn, nn.MultiheadAttention):
                    attn_mask = generate_tgt_mask(x.shape[1]).to(
                        x.device
                    )  # (s, s), in this case binary attn_mask is allowed

        if attn_mask is not None and key_padding_mask is not None:
            attn_mask = combine_key_pad_and_attn(
                attn_mask, key_padding_mask, self.nhead
            )  # (b, nh, s, s)
            key_padding_mask = None
        if (
            isinstance(self.self_attn, nn.MultiheadAttention)
            and attn_mask is not None
        ):
            if attn_mask.ndim == 3:  # (b, s, s) -> (b*nh, s, s)
                attn_mask = attn_mask.repeat_interleave(self.nhead, dim=0)
            elif attn_mask.ndim == 4:  # (b, nh, s, s) -> (b*nh, s, s)
                attn_mask = attn_mask.view(-1, x.shape[-2], x.shape[-2])
            # attn_mask can also be (s, s) but
            # nn.MultiHeadAttention backends can handle this

        position_kwargs = (
            {}
            if isinstance(self.self_attn, nn.MultiheadAttention)
            else {"positions": positions}
        )
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=pass_causal,
            **position_kwargs,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
    """A custom Transformer decoder layer.

    Parameters
    ----------
    d_model : int
        The number of expected features in the input.
    nhead : int
        The number of heads in the MultiHeadAttention module.
    dim_feedforward : int, optional
        The dimension of the feedforward network model.
    dropout : float, optional
        The dropout value.
    activation : str or callable, optional
        The activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: "relu"
    layer_norm_eps : float, optional
        The eps value in layer normalization components.
    norm_first : bool, optional
        If True, layer norm is done prior to attention and feedforward
        operations (pre-norm). Otherwise it's done after (post-norm).
    batch_first : bool, optional
        Placeholder for API compatibility with
        `torch.nn.TransformerDecoderLayer`.
        Should always be True.
    attention_backend : str, optional
        Attention implementation: "sdpa" (default) or "native".
    rotary_embedding : RotaryEmbedding, optional
        Rotary position embedding module to apply to Q and K in self-attention.
        If None, no rotary embeddings are used.
        Note: Only applied to self-attention, not cross-attention.
    enable_sdpa_math : bool, optional
        If True, enable SDPA math kernel when using "sdpa" backend.
        Default: True
    enable_sdpa_mem_efficient : bool, optional
        If True, enable SDPA memory efficient kernel when using "sdpa" backend.
        Default: True
    enable_sdpa_flash_attention : bool, optional
        If True, enable SDPA flash attention kernel when using "sdpa" backend.
        Default: True

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | callable = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        batch_first: bool = True,
        attention_backend: str = "sdpa",
        rotary_embedding: nn.Module | None = None,
        enable_sdpa_math: bool = True,
        enable_sdpa_mem_efficient: bool = True,
        enable_sdpa_flash_attention: bool = True,
    ) -> None:
        """Initialize a TransformerDecoderLayer."""
        super().__init__()

        if not batch_first:
            raise ValueError(
                "TransformerDecoderLayer requires batch_first=True"
            )

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.norm_first = norm_first
        self.attention_backend = attention_backend

        attn_args = {
            "embed_dim": d_model,
            "num_heads": nhead,
            "dropout": dropout,
            "batch_first": True,
        }

        if attention_backend == "native":
            if rotary_embedding is not None:
                raise ValueError(
                    "Rotary embeddings are not supported with the native "
                    "attention backend."
                )

            self.self_attn = nn.MultiheadAttention(
                **attn_args,
            )
        elif attention_backend == "sdpa":
            self.self_attn = MultiheadAttention(
                **attn_args,
                rotary_embedding=rotary_embedding,
                enable_sdpa_math=enable_sdpa_math,
                enable_sdpa_mem_efficient=enable_sdpa_mem_efficient,
                enable_sdpa_flash_attention=enable_sdpa_flash_attention,
            )
        else:
            raise ValueError(
                '`attention_backend` should be one of "sdpa" or "native"'
            )

        if attention_backend == "native":
            self.multihead_attn = nn.MultiheadAttention(
                **attn_args,
            )
        elif attention_backend == "sdpa":
            self.multihead_attn = MultiheadAttention(
                **attn_args,
                enable_sdpa_math=enable_sdpa_math,
                enable_sdpa_mem_efficient=enable_sdpa_mem_efficient,
                enable_sdpa_flash_attention=enable_sdpa_flash_attention,
            )
        else:
            raise ValueError(
                '`attention_backend` should be one of "sdpa" or "native"'
            )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation = nn.functional.relu
        elif activation == "gelu":
            self.activation = nn.functional.gelu
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError(
                f"activation should be 'relu', 'gelu', or a callable, "
                f"not {activation}"
            )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        tgt_positions: Tensor | None = None,
    ) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Parameters
        ----------
        tgt : Tensor
            (batch_size, tgt_seq_len, d_model).
            The sequence to the decoder layer.
        memory : Tensor
            (batch_size, src_seq_len, d_model).
            The sequence from the last layer of the encoder.
        tgt_mask : Tensor, optional
            If specified, pairwise additive (float) mask
            Shape:
                - (tgt_seq_len, tgt_seq_len),
                - (batch_size, tgt_seq_len, tgt_seq_len), or
                - (batch_size, nhead, tgt_seq_len, tgt_seq_len)
        memory_mask : Tensor, optional
            If specified, pairwise additive (float) mask
            Shape:
                - (tgt_seq_len, src_seq_len),
                - (batch_size, tgt_seq_len, src_seq_len), or
                - (batch_size, nhead, tgt_seq_len, src_seq_len)
        tgt_key_padding_mask : Tensor, optional
            If specified, should be a binary mask of shape
            (batch_size, tgt_seq_len) indicating which elements
            should participate in attention. `True` values indicate
            positions that should not participate in attention
            (e.g., padded elements).
        memory_key_padding_mask : Tensor, optional
            If specified, should be a binary mask of shape
            (batch_size, src_seq_len) indicating which elements
            should participate in attention. `True` values indicate
            positions that should not participate in attention
            (e.g., padded elements).
        tgt_is_causal : bool, optional
            If True, applies a causal mask to the sequence to the
            decoder layer.
        memory_is_causal : bool, optional
            If True, applies a causal mask during cross_attention
            from memory to targets.
        tgt_positions : Tensor, optional
            Position values for rotary embeddings in self-attention on tgt.
            Ignored when RotaryEmbedding is not used.
            Not used for cross-attention. If None, uses integer positions.
            Shape: (batch, tgt_seq_len) or (tgt_seq_len,).

        Returns
        -------
        Tensor
            The output of the decoder layer.

        """
        if tgt_mask is not None:
            _validate_attn_mask(
                tgt_mask,
                tgt,
                self.nhead,
                "tgt_mask",
                tgt.size(1),
            )
        if tgt_key_padding_mask is not None:
            _validate_key_padding_mask(
                tgt_key_padding_mask,
                tgt.size(0),
                tgt.size(1),
                "tgt_key_padding_mask",
            )
        if memory_mask is not None:
            _validate_attn_mask(
                memory_mask,
                memory,
                self.nhead,
                "memory_mask",
                tgt.size(1),
                memory.size(1),
            )
        if memory_key_padding_mask is not None:
            _validate_key_padding_mask(
                memory_key_padding_mask,
                memory.size(0),
                memory.size(1),
                "memory_key_padding_mask",
            )

        if self.norm_first:
            # Pre-norm architecture
            tgt = tgt + self._sa_block(
                self.norm1(tgt),
                tgt_mask,
                tgt_key_padding_mask,
                tgt_is_causal,
                tgt_positions,
            )
            tgt = tgt + self._mha_block(
                self.norm2(tgt),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            # Post-norm architecture (default for PyTorch)
            tgt = self.norm1(
                tgt
                + self._sa_block(
                    tgt,
                    tgt_mask,
                    tgt_key_padding_mask,
                    tgt_is_causal,
                    tgt_positions,
                )
            )
            tgt = self.norm2(
                tgt
                + self._mha_block(
                    tgt,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                )
            )
            tgt = self.norm3(tgt + self._ff_block(tgt))

        return tgt

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
        positions: Tensor | None = None,
    ) -> Tensor:
        """Does self-attention with argument interpretation.

        When causal and attn_mask is given: combine, then don't pass
        is_causal to MultiHeadAttention.
        When causal and no attn_mask: if native MultiHeadAttention,
        then pass is_causal with an explicit mask, else just
        pass is_causal.

        In addition, combines attn_mask with key_padding_mask
        if both are passed.
        """
        pass_causal = False
        if is_causal:
            if attn_mask is not None:
                causal_mask = generate_tgt_mask(x.shape[-2]).to(x.device)
                attn_mask = attn_mask.masked_fill(
                    causal_mask, -float("inf")
                )  # broadcasts to (s, s), (b, s, s) or (b, nh, s, s)
                pass_causal = False
            else:
                pass_causal = True
                if isinstance(self.self_attn, nn.MultiheadAttention):
                    attn_mask = generate_tgt_mask(x.shape[1]).to(
                        x.device
                    )  # (s, s), in this case binary attn_mask is allowed

        if attn_mask is not None and key_padding_mask is not None:
            attn_mask = combine_key_pad_and_attn(
                attn_mask, key_padding_mask, self.nhead
            )  # (b, nh, s, s)
            key_padding_mask = None

        if (
            isinstance(self.self_attn, nn.MultiheadAttention)
            and attn_mask is not None
        ):
            if attn_mask.ndim == 3:  # (b, s, s) -> (b*nh, s, s)
                attn_mask = attn_mask.repeat_interleave(self.nhead, dim=0)
            elif attn_mask.ndim == 4:  # (b, nh, s, s) -> (b*nh, s, s)
                attn_mask = attn_mask.view(-1, x.shape[-2], x.shape[-2])
            # attn_mask can also be (s, s) but
            # nn.MultiHeadAttention backends can handle this

        position_kwargs = (
            {}
            if isinstance(self.self_attn, nn.MultiheadAttention)
            else {"positions": positions}
        )
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=pass_causal,
            **position_kwargs,
        )[0]
        return self.dropout1(x)

    # multihead (cross) attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        """Performs cross-attention (multihead attention with memory).

        This method applies multihead attention where `x` is the target (query)
        and `mem` is the memory (key and value), as in cross-attention.
        In addition, combines `attn_mask` with `key_padding_mask`
        if both are provided.
        """
        if is_causal is not False:
            raise ValueError("Causal cross-attention is not supported.")

        if attn_mask is not None and key_padding_mask is not None:
            attn_mask = combine_key_pad_and_attn(
                attn_mask, key_padding_mask, self.nhead
            )  # (b, nh, tgt, mem)
            key_padding_mask = None

        if (
            isinstance(self.self_attn, nn.MultiheadAttention)
            and attn_mask is not None
        ):
            if attn_mask.ndim == 3:  # (b, tgt, mem) -> (b*nh, tgt, mem)
                attn_mask = attn_mask.repeat_interleave(self.nhead, dim=0)
            elif attn_mask.ndim == 4:  # (b, nh, tgt, mem) -> (b*nh, tgt, mem)
                attn_mask = attn_mask.view(-1, x.shape[-2], mem.shape[-2])
            # attn_mask can also be (tgt, mem) but
            # nn.MultiHeadAttention backends can handle this

        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, n):
    # FIXME: copy.deepcopy() is not defined on nn.module,
    # from PyTorch source code
    import copy

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerEncoder(nn.Module):
    """Custom Transformer encoder.

    A stack of N encoder layers, replicates `torch.nn.TransformerEncoder`.

    Parameters
    ----------
    encoder_layer : TransformerEncoderLayer
        An instance of the TransformerEncoderLayer class.
    num_layers : int
        The number of sub-encoder-layers in the encoder.
    norm : nn.Module, optional
        The layer normalization component (optional).

    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        """Initialize a TransformerEncoder."""
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
        positions: Tensor | None = None,
    ) -> Tensor:
        """Pass the input through the encoder layers in turn.

        Parameters
        ----------
        src : Tensor
            (batch_size, seq_len, d_model).
        mask : Tensor, optional
            If specified, pairwise additive (float) mask
            Shape:
                - (batch_size, seq_len, seq_len) or
                - (batch_size, nhead, seq_len, seq_len)
        src_key_padding_mask : Tensor, optional
            If specified, should be a binary mask of shape
            (batch_size, seq_len) indicating which elements should
            participate in attention. `True` values indicate positions
            that should not participate in attention (e.g., padded elements).
        is_causal : bool, optional
            If True, applies a causal mask.
        positions : Tensor, optional
            Position values for rotary embeddings.
            If None, uses integer positions.
            Ignored when RotaryEmbedding is not used.
            Shape: (batch, seq_len) or (seq_len,).

        Returns
        -------
        Tensor of shape (batch_size, seq_len, d_model)
            The output of the encoder.

        """
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                positions=positions,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """Custom Transformer decoder.

    A stack of N decoder layers. This implementation exactly replicates the
    behavior of `torch.nn.TransformerDecoder` and serves as a foundation for
    future extensions.

    Parameters
    ----------
    decoder_layer : TransformerDecoderLayer
        An instance of the TransformerDecoderLayer class.
    num_layers : int
        The number of sub-decoder-layers in the decoder.
    norm : nn.Module, optional
        The layer normalization component (optional).

    """

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        """Initialize a TransformerDecoder."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        tgt_positions: Tensor | None = None,
    ) -> Tensor:
        """Pass the inputs (and masks) through the decoder layers in turn.

        Parameters
        ----------
        tgt : Tensor of shape (batch_size, tgt_seq_len, d_model)
            The sequence to the decoder.
        memory : Tensor of shape (batch_size, src_seq_len, d_model)
            The sequence from the last layer of the encoder.
        tgt_mask : Tensor, optional
            If specified, pairwise additive (float) mask
            during target self-attention
            Shape:
                - (batch_size, tgt_seq_len, tgt_seq_len) or
                - (batch_size, nhead, tgt_seq_len, tgt_seq_len)
        memory_mask : Tensor, optional
            If specified, pairwise additive (float) mask
            Shape:
                - (batch_size, tgt_seq_len, src_seq_len) or
                - (batch_size, nhead, tgt_seq_len, src_seq_len)
        tgt_key_padding_mask : Tensor, optional
            If specified, should be a binary mask of shape
            (batch_size, tgt_seq_len) indicating which elements
            should participate in attention. `True` values indicate
            positions that should not participate in attention
            (e.g., padded elements).
        memory_key_padding_mask : Tensor, optional
            If specified, should be a binary mask of shape
            (batch_size, src_seq_len) indicating which elements
            should participate in attention. `True` values indicate
            positions that should not participate in attention
            (e.g., padded elements).
        tgt_is_causal : bool, optional
            If True, applies a causal mask to the sequence to the
            decoder layer.
        memory_is_causal : bool, optional
            If True, applies a causal mask during cross_attention from
            memory to targets.
        tgt_positions : Tensor, optional
            Position values for rotary embeddings in self-attention on tgt.
            Ignored when RotaryEmbedding is not used.
            Not used for cross-attention. If None, uses integer positions.
            Shape: (batch, tgt_seq_len) or (tgt_seq_len,).

        Returns
        -------
        Tensor of shape (batch_size, tgt_seq_len, d_model)
            The output of the decoder.

        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
                tgt_positions=tgt_positions,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
