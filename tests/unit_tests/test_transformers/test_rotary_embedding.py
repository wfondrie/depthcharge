"""Test RotaryEmbedding implementation."""

import numpy as np
import pytest
import torch

from depthcharge.encoders.rotary import RotaryEmbedding
from depthcharge.transformers import (
    AnalyteTransformerDecoder,
    AnalyteTransformerEncoder,
    SpectrumTransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


def _attn(q, k):
    head_dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
    return scores


@pytest.mark.parametrize("shift", [0, 5, 10, 100])
@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
def test_shift_invariance(shift, batch_size, seq_len):
    """Test that RoPE produces shift-invariant attention patterns."""
    head_dim = 16
    rope = RotaryEmbedding(head_dim=head_dim)

    torch.manual_seed(42)
    q = torch.randn(batch_size, 8, seq_len, head_dim)
    torch.manual_seed(43)
    k = torch.randn(batch_size, 8, seq_len, head_dim)

    pos_0 = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    q_rot_0, k_rot_0 = rope(q, k, positions=pos_0)

    q_rot_shifted, k_rot_shifted = rope(q, k, positions=pos_0 + shift)

    torch.testing.assert_close(
        _attn(q_rot_0, k_rot_0),
        _attn(q_rot_shifted, k_rot_shifted),
        rtol=1e-3,
        atol=1e-5,
        msg=f"Attention patterns differ with shift={shift}, seq_len={seq_len}",
    )


@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
def test_encoder(batch_size, seq_len):
    """Test rotary within TransformerEncoder."""
    torch.manual_seed(42)
    custom_layer = TransformerEncoderLayer(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        batch_first=True,
        dropout=0.0,
        norm_first=True,
        attention_backend="sdpa",
        activation="relu",
        rotary_embedding=RotaryEmbedding(head_dim=256 // 8),
    )
    custom_transformer = TransformerEncoder(
        custom_layer,
        num_layers=2,
    )

    torch.manual_seed(43)
    x = torch.randn(batch_size, seq_len, 256)

    result_spec_pos = custom_transformer(
        src=x,
        positions=torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
    )

    result_infer_pos = custom_transformer(
        src=x,
        mask=None,
    )
    torch.testing.assert_close(
        result_infer_pos,
        result_spec_pos,
        rtol=1e-3,
        atol=1e-5,
    )


@pytest.mark.parametrize("batch_size,seq_len", [(2, 10), (4, 20), (1, 5)])
def test_decoder(batch_size, seq_len):
    """Test rotary within TransformerDecoder."""
    torch.manual_seed(42)
    custom_layer = TransformerDecoderLayer(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        batch_first=True,
        dropout=0.0,
        norm_first=True,
        attention_backend="sdpa",
        activation="relu",
        rotary_embedding=RotaryEmbedding(head_dim=256 // 8),
    )
    custom_transformer = TransformerDecoder(
        custom_layer,
        num_layers=2,
    )

    torch.manual_seed(43)
    x = torch.randn(batch_size, seq_len, 256)
    torch.manual_seed(44)
    mem = torch.randn(batch_size, seq_len + 5, 256)
    result_spec_pos = custom_transformer(
        tgt=x,
        memory=mem,
        tgt_positions=torch.arange(seq_len)
        .unsqueeze(0)
        .expand(batch_size, -1),
    )

    result_infer_pos = custom_transformer(
        tgt=x,
        memory=mem,
    )
    torch.testing.assert_close(
        result_infer_pos,
        result_spec_pos,
        rtol=1e-3,
        atol=1e-5,
    )


def test_spectrumencoder():
    """Test if a SpectrumTransformerEncoder with Rotary Embeddings runs."""
    torch.manual_seed(42)
    model = SpectrumTransformerEncoder(
        d_model=128,
        nhead=8,
        n_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        rotary_embedding=RotaryEmbedding(head_dim=16),
    )
    torch.manual_seed(43)
    mz_array = torch.rand(8, 15)
    torch.manual_seed(44)
    int_array = torch.rand(8, 15) * 1_000 + 50

    seqlens = [2, 12, 9, 4, 5, 10, 1, 7]
    for i, s_i in enumerate(seqlens):  # var len seq
        mz_array[i, s_i:] = 0
        int_array[i, s_i:] = 0

    _ = model(mz_array, int_array)


def test_analyteencoder():
    """Test if a AnalyteTransformerEncoder with Rotary Embeddings runs."""
    torch.manual_seed(42)
    model = AnalyteTransformerEncoder(
        n_tokens=100,
        d_model=128,
        nhead=8,
        n_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        rotary_embedding=RotaryEmbedding(head_dim=16),
        padding_int=0,
    )
    torch.manual_seed(43)
    tokens = torch.randint(low=1, high=100, size=(8, 15))
    seqlens = [2, 12, 9, 4, 5, 10, 1, 7]
    for i, s_i in enumerate(seqlens):  # var len seq
        tokens[i, s_i:] = 0

    _ = model(tokens)


def test_analytedecoder():
    """Test if a AnalyteTransformerDecoder with Rotary Embeddings runs."""
    torch.manual_seed(42)
    model = AnalyteTransformerDecoder(
        n_tokens=100,
        d_model=128,
        nhead=8,
        n_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        rotary_embedding=RotaryEmbedding(head_dim=16),
        padding_int=0,
    )
    torch.manual_seed(43)

    tokens = torch.randint(low=1, high=100, size=(8, 15))
    seqlens = [2, 12, 9, 4, 5, 10, 1, 7]
    for i, s_i in enumerate(seqlens):  # var len seq
        tokens[i, s_i:] = 0

    mems = torch.randn(8, 20, 128)
    seqlens = torch.tensor([5, 7, 5, 9, 4, 15, 18, 16])
    mem_pad_mask = torch.arange(20).unsqueeze(0) >= seqlens.unsqueeze(1)

    _ = model(tokens, memory=mems, memory_key_padding_mask=mem_pad_mask)
