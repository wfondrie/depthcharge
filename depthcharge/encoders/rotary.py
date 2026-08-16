"""Rotary Position Embeddings (RoPE) for Transformers."""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat


class RotaryEmbedding(nn.Module):
    """Rotary embedding as in RoFormer / RoPE.

    Applies rotary positional embeddings to query and key tensors in attention.
    Unlike additive positional encodings, RoPE rotates the Q and K vectors
    based on their position, preserving relative position information in the
    attention computation.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head.
    min_wavelength : float, optional
        The minimum wavelength to use.
    max_wavelength : float, optional
        The maximum wavelength to use.

    """

    def __init__(
        self,
        head_dim: int,
        min_wavelength: float = 2 * np.pi,
        max_wavelength: float = 20_000 * np.pi,
    ) -> None:
        """Initialize RotaryEmbedding."""
        super().__init__()
        self.head_dim = head_dim

        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength
        thetas = (
            1.0
            / base
            / (scale ** (torch.arange(0, head_dim, 2).float() / head_dim))
        )
        self.register_buffer("thetas", thetas)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k : torch.Tensor
            Key tensor of shape (batch, num_heads, seq_len, head_dim)
        positions : torch.Tensor, optional
            Position values for each element in the sequence.
            - If None: Use integer positions torch.arange(seq_len)
            - Shape: (batch, seq_len) for per-sample positions
            Examples: integer positions [0,1,2,..],
            or m/z values [100.5, 150.2, ...]
            Default: None

        Returns
        -------
        q_rotated : torch.Tensor
            Query with rotary embeddings applied, same shape as input
        k_rotated : torch.Tensor
            Key with rotary embeddings applied, same shape as input

        """
        pos_to_use = self._default_pos(q) if positions is None else positions
        if pos_to_use.ndim == 2:
            pos_to_use = pos_to_use[:, None].expand(-1, q.size(1), -1)

        sin, cos = self._get_rotations(pos_to_use, self.thetas)
        q_rot = q * cos.to(q) + self._rotate_every_two(q) * sin.to(q)
        k_rot = k * cos.to(k) + self._rotate_every_two(k) * sin.to(k)

        return q_rot, k_rot

    @staticmethod
    def _default_pos(x):
        return torch.arange(x.size(-2), device=x.device).expand(*x.shape[:-1])

    @staticmethod
    def _get_rotations(pos, thetas):
        mthetas = pos[..., None] * thetas  # (..., seq_len, head_dim/2)

        sin, cos = (
            repeat(t, "b ... h  -> b ... (h j)", j=2).to(thetas)
            for t in (mthetas.sin(), mthetas.cos())
        )
        return sin, cos

    @staticmethod
    def _rotate_every_two(x):
        x = x.clone()
        x = rearrange(x, "... (d j) -> ... d j", j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d j -> ... (d j)")
