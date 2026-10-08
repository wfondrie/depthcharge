"""Custom attention mechanisms for depthcharge transformers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..utils import generate_tgt_mask


class MultiheadAttention(nn.Module):
    """Multi-head attention using scaled_dot_product_attention.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model.
    num_heads : int
        Number of parallel attention heads. embed_dim must be divisible
        by num_heads.
    dropout : float, optional
        Dropout probability on attention weights. Default: 0.0
    bias : bool, optional
        If True, add bias to input/output projections. Default: True
    batch_first : bool, optional
        Placeholder for API compatibility.
        Should always be True.
    rotary_embedding : RotaryEmbedding, optional
        Rotary position embedding module to apply to Q and K.
        If None, no rotary embeddings are used. Default: None
    enable_sdpa_math : bool, optional
        If True, enable SDPA math kernel. Default: True
    enable_sdpa_mem_efficient : bool, optional
        If True, enable SDPA memory efficient kernel. Default: True
    enable_sdpa_flash_attention : bool, optional
        If True, enable SDPA flash attention kernel. Default: True

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        rotary_embedding: nn.Module | None = None,
        enable_sdpa_math: bool = True,
        enable_sdpa_mem_efficient: bool = True,
        enable_sdpa_flash_attention: bool = True,
    ) -> None:
        """Initialize MultiheadAttention."""
        super().__init__()

        if batch_first is not True:
            raise ValueError("MultiheadAttention requires batch_first=True")

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.head_dim = embed_dim // num_heads
        self.rotary_embedding = rotary_embedding

        self.context_manager = []
        if enable_sdpa_math:
            self.context_manager.append(SDPBackend.MATH)
        if enable_sdpa_flash_attention:
            self.context_manager.append(SDPBackend.FLASH_ATTENTION)
        if enable_sdpa_mem_efficient:
            self.context_manager.append(SDPBackend.EFFICIENT_ATTENTION)

        self.use_context_manager = not all(
            [
                enable_sdpa_math,
                enable_sdpa_flash_attention,
                enable_sdpa_mem_efficient,
            ]
        )

        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * embed_dim, embed_dim)
        )
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass of multi-head attention.

        Parameters
        ----------
        query : Tensor
            Query embeddings. Shape: (batch, tgt_len, embed_dim).
        key : Tensor
            Key embeddings. Shape: (batch, src_len, embed_dim).
        value : Tensor or NestedTensor
            Value embeddings. Shape: (batch, src_len, embed_dim).
        key_padding_mask : Tensor, optional
            If specified, a mask of shape (batch, src_len) indicating which
            elements should participate in attention. `True` values indicate
            positions that should not participate in attention
            (e.g., padded elements).
        need_weights : bool, optional
            Placeholder for compatibility with `torch.nn.MultiheadAttention`.
            This module always returns None for attention weights.
        attn_mask : Tensor, optional
            If specified, pairwise additive (float) mask.
            Shape:
                - (tgt_seq_len, src_len),
                - (batch_size, tgt_seq_len, src_len), or
                - (batch_size, nhead, tgt_len, src_len).
        is_causal : bool, optional
            If True, apply causal masking (lower triangular). Default: False
        positions : Tensor, optional
            Position values for rotary embeddings.
            Only used if rotary_embedding is configured.
            If None, uses integer positions. Can be:
            - (batch, seq_len) for per-sample positions (e.g., m/z values)
            - (seq_len,) for shared positions
            Default: None.

        Returns
        -------
        attn_output : Tensor or NestedTensor
            Attention output. Same layout as inputs.
        attn_weights : None
            Placeholder for compatibility with `torch.nn.MultiHeadAttention`.
            Always None.

        """
        if attn_mask is not None and key_padding_mask is not None:
            raise ValueError(
                "attn_mask and key_padding_mask can not be simultaneously "
                "given. Use depthcharge.utils.combine_key_pad_and_attn to "
                "combine masks."
            )
        # Apply input projections
        if query is key and key is value:  # self-attention case
            qkv = nn.functional.linear(
                query, self.in_proj_weight, self.in_proj_bias
            )
            q, k, v = qkv.chunk(3, dim=-1)
        else:  # cross-attention case
            w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
            if self.in_proj_bias is not None:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
            else:
                b_q, b_k, b_v = None, None, None

            q = nn.functional.linear(query, w_q, b_q)
            k = nn.functional.linear(key, w_k, b_k)
            v = nn.functional.linear(value, w_v, b_v)

        # [bsz, seqlen, embed_dim] -> [bsz, nhead, seqlen, headdim]
        q = q.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        k = k.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

        # Apply rotary embeddings
        if self.rotary_embedding is not None:
            q, k = self.rotary_embedding(q, k, positions=positions)

        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask[:, None, None, :]

        if attn_mask is not None and attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None]

        if attn_mask is not None and is_causal:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.logical_and(
                    attn_mask,
                    ~generate_tgt_mask(attn_mask.size(-1)).to(
                        attn_mask.device
                    ),
                )
                is_causal = False
            else:
                causal_mask = generate_tgt_mask(q.shape[-2]).to(
                    attn_mask.device
                )
                attn_mask = attn_mask.masked_fill(
                    causal_mask, -float("inf")
                )  # broadcasts to (b, s, s) or (b, nh, s, s)
                is_causal = False

        attn_output = self._sdpa_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )

        # [bsz, nhead, seqlen, headdim] -> [bsz, seqlen, embed_dim]
        attn_output = attn_output.transpose(1, 2).flatten(-2, -1)

        attn_output = self.out_proj(attn_output)
        return attn_output, None

    def _sdpa_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        if self.use_context_manager:
            with sdpa_kernel(self.context_manager):
                attn_output = nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=is_causal,
                )
        else:
            attn_output = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )

        return attn_output
