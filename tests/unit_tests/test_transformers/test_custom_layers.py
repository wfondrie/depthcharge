"""Test custom transformers for numerical equivalence with PyTorch defaults."""

import pytest
import torch
import torch.nn as nn

from depthcharge.transformers.layers import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from depthcharge.utils import generate_tgt_mask


@pytest.mark.parametrize(
    "d_model,nhead,dim_feedforward,dropout",
    [
        (
            128,
            8,
            1024,
            0.0,
        ),  # dropout always 0 because PyTorch's SDPA is non-deterministic
        (64, 4, 256, 0.0),
    ],
)
@pytest.mark.parametrize("batch_size,seq_len", [(4, 20), (3, 15)])
@pytest.mark.parametrize("pad_keys", [True, False])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("backend", ["sdpa", "native"])
@pytest.mark.parametrize("activation", ["relu", "gelu", lambda x: x + 1])
@pytest.mark.parametrize("test_float_attn", [True, False])
def test_encoder_equivalence(
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    batch_size,
    seq_len,
    pad_keys,
    norm_first,
    is_causal,
    backend,
    activation,
    test_float_attn,
):
    """Test numerical equivalence with PyTorch TransformerEncoderLayer."""
    torch.manual_seed(42)
    pytorch_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        activation=activation,
    )
    pytorch_transformer = nn.TransformerEncoder(
        pytorch_layer,
        num_layers=2,
    )

    torch.manual_seed(42)
    custom_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        attention_backend=backend,
        activation=activation,
    )
    custom_transformer = TransformerEncoder(
        custom_layer,
        num_layers=2,
    )

    custom_transformer.load_state_dict(pytorch_transformer.state_dict())

    # Create fixed-length dense tensors
    torch.manual_seed(43)
    src_pytorch = torch.randn(batch_size, seq_len, d_model)
    src_custom = src_pytorch.clone()
    src_pytorch.requires_grad = True
    src_custom.requires_grad = True

    if pad_keys:
        seq_lens = seq_len - torch.arange(1, batch_size + 1)
        src_key_padding_mask = torch.arange(seq_len).unsqueeze(
            0
        ) >= seq_lens.unsqueeze(1)
    else:
        src_key_padding_mask = None

    if test_float_attn:
        attn_mask = torch.randn(seq_len, seq_len)
        if is_causal:
            # skip these cases as default pytorch transformer
            # cannot handle both causal and float attn_mask.
            return None
    else:
        attn_mask = None

    torch.manual_seed(44)
    pytorch_output = pytorch_transformer(
        src_pytorch,
        is_causal=is_causal,
        mask=(generate_tgt_mask(seq_len) if is_causal else attn_mask),
        src_key_padding_mask=src_key_padding_mask,
    )
    torch.manual_seed(44)
    custom_output = custom_transformer(
        src_custom,
        is_causal=is_causal,
        mask=attn_mask,
        src_key_padding_mask=src_key_padding_mask,
    )

    if pad_keys:
        # In some cases the padded tokens are handled differently,
        # this does not matter, as long as unpadded tokens are equal.
        pytorch_output = pytorch_output[~src_key_padding_mask]
        custom_output = custom_output[~src_key_padding_mask]

    torch.testing.assert_close(
        custom_output,
        pytorch_output,
        rtol=1e-3,
        atol=1e-5,
        msg="Custom layer output does not match PyTorch layer output",
    )

    # Backward pass
    pytorch_loss = pytorch_output.sum()
    custom_loss = custom_output.sum()

    pytorch_loss.backward()
    custom_loss.backward()

    # Check gradients match
    torch.testing.assert_close(
        src_custom.grad,
        src_pytorch.grad,
        rtol=1e-3,
        atol=1e-5,
        msg="Gradients do not match",
    )


@pytest.mark.parametrize(
    "d_model,nhead,dim_feedforward,dropout",
    [
        (
            128,
            8,
            1024,
            0.0,
        ),  # dropout always 0 because PyTorch's SDPA is non-deterministic
        (64, 4, 256, 0.0),
    ],
)
@pytest.mark.parametrize("batch_size,seq_len", [(4, 20), (3, 15)])
@pytest.mark.parametrize("pad_keys", [True, False])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("backend", ["sdpa", "native"])
@pytest.mark.parametrize("activation", ["relu", "gelu", lambda x: x + 1])
@pytest.mark.parametrize("test_float_attn", [True, False])
def test_decoder_equivalence(
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    batch_size,
    seq_len,
    pad_keys,
    norm_first,
    is_causal,
    backend,
    activation,
    test_float_attn,
):
    """Test numerical equivalence with PyTorch TransformerDecoderLayer."""
    torch.manual_seed(42)
    pytorch_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        activation=activation,
    )
    pytorch_transformer = nn.TransformerDecoder(
        pytorch_layer,
        num_layers=2,
    )

    torch.manual_seed(42)
    custom_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
        dropout=dropout,
        norm_first=norm_first,
        attention_backend=backend,
        activation=activation,
    )
    custom_transformer = TransformerDecoder(
        custom_layer,
        num_layers=2,
    )

    custom_transformer.load_state_dict(pytorch_transformer.state_dict())

    torch.manual_seed(43)
    tgt_pytorch = torch.randn(batch_size, seq_len, d_model)
    tgt_custom = tgt_pytorch.clone()
    tgt_pytorch.requires_grad = True
    tgt_custom.requires_grad = True

    # Memory tensor for cross-attention with seq_len + 5
    memory = torch.randn(batch_size, seq_len + 5, d_model)

    if pad_keys:
        seq_lens = seq_len - torch.arange(1, batch_size + 1)
        tgt_key_padding_mask = torch.arange(seq_len).unsqueeze(
            0
        ) >= seq_lens.unsqueeze(1)
        seq_lens = seq_len + 5 - torch.arange(1, batch_size + 1)
        memory_key_padding_mask = torch.arange(seq_len + 5).unsqueeze(
            0
        ) >= seq_lens.unsqueeze(1)
    else:
        tgt_key_padding_mask = None
        memory_key_padding_mask = None

    if test_float_attn:
        tgt_mask = torch.randn(seq_len, seq_len)
        memory_mask = torch.randn(seq_len, seq_len + 5)
        if is_causal:
            # skip these cases as default pytorch transformer
            # cannot handle both causal and float attn_mask.
            return None
    else:
        tgt_mask = None
        memory_mask = None

    torch.manual_seed(44)
    pytorch_output = pytorch_transformer(
        tgt_pytorch,
        memory,
        tgt_mask=(generate_tgt_mask(seq_len) if is_causal else tgt_mask),
        memory_mask=memory_mask,
        tgt_is_causal=is_causal,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    torch.manual_seed(44)
    custom_output = custom_transformer(
        tgt_custom,
        memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        tgt_is_causal=is_causal,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )

    if pad_keys:
        # In some cases the padded tokens are handled differently,
        # this does not matter, as long as unpadded tokens are equal.
        pytorch_output = pytorch_output[~tgt_key_padding_mask]
        custom_output = custom_output[~tgt_key_padding_mask]

    torch.testing.assert_close(
        custom_output,
        pytorch_output,
        rtol=1e-3,
        atol=1e-5,
        msg="Custom layer output does not match PyTorch layer output",
    )

    # Backward pass
    pytorch_loss = pytorch_output.sum()
    custom_loss = custom_output.sum()

    pytorch_loss.backward()
    custom_loss.backward()

    # Check gradients match
    torch.testing.assert_close(
        tgt_custom.grad,
        tgt_pytorch.grad,
        rtol=1e-3,
        atol=1e-5,
        msg="Gradients do not match",
    )
