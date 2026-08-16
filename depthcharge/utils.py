"""Common utility functions."""

from typing import Any

import polars as pl
import torch


def listify(obj: Any) -> list[Any]:  # noqa: ANN401
    """Turn an object into a list, but don't split strings."""
    try:
        invalid = [str, pl.DataFrame, pl.LazyFrame]
        if any(isinstance(obj, c) for c in invalid):
            raise TypeError

        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)


def generate_tgt_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.

    """
    return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)


def combine_key_pad_and_attn(
    attn_mask: torch.Tensor,
    key_padding_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Combine float attn_mask and binary key_padding_mask.

    Parameters
    ----------
    attn_mask : torch.Tensor
        float mask of shape (S1,S2), (N, S1, S2) or (N, Nheads, S1, S2)
    key_padding_mask: torch.Tensor
        binary mask of shape (N, S2), where True = masked
    num_heads: int
        number of heads.

    Returns
    -------
    torch.Tensor
        Combined attention mask of shape (N, num_heads, S1, S2).

    """
    n, s2 = key_padding_mask.shape
    s1 = attn_mask.size(-2)

    kpm = key_padding_mask[:, None, None, :].expand(
        n, num_heads, s1, s2
    )  # (n, nh, s1, s2)
    kpm_float = torch.zeros_like(
        kpm, dtype=attn_mask.dtype, device=key_padding_mask.device
    )
    kpm_float[kpm] = float("-inf")

    if attn_mask.ndim == 2:  # (s1, s2) -> (n, nh, s1, s2)
        am = attn_mask[None, None, :, :].expand(n, num_heads, s1, s2)
    elif attn_mask.ndim == 3:  # (n, s1, s2) -> (n, nh, s1, s2)
        am = attn_mask[:, None].expand(n, num_heads, s1, s2)
    elif attn_mask.ndim == 4:  # (n, nh, s1, s2)
        am = attn_mask

    out = am + kpm_float
    return out
