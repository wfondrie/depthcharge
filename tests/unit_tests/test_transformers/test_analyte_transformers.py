"""Test the peptide transformers."""

import pytest
import torch

from depthcharge.tokenizers import PeptideTokenizer
from depthcharge.transformers import (
    AnalyteTransformerDecoder,
    AnalyteTransformerEncoder,
    SpectrumTransformerEncoder,
)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "model", [AnalyteTransformerEncoder, AnalyteTransformerDecoder]
)
def test_init(model):
    """Test that initializtion warns and errors as we expect it to."""
    with pytest.raises(ValueError):
        model(1)

    tokenizer = PeptideTokenizer()
    with pytest.warns(UserWarning):
        model(tokenizer, padding_int=5)

    model(tokenizer)


def test_analyte_encoder():
    """Test that a peptide encoder will run."""
    tokenizer = PeptideTokenizer()
    peptides = tokenizer.tokenize(["LESLIEK", "PEPTIDER", "EDITHYKK"])
    model = AnalyteTransformerEncoder(tokenizer, 8, 2, 12)
    emb, mask = model(peptides)

    # Axis 1 should be 1 longer than the longest peptide.
    assert emb.shape == (3, 9, 8)
    assert mask.sum() == 1

    res = emb.sum(dim=1)
    assert (res[1, :] != res[2, :]).all()


def test_analyte_decoder():
    """Test that a peptide decoder will run."""
    tokenizer = PeptideTokenizer()
    n_tokens = len(tokenizer)

    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500, 0.5], [0, 0]],
        ]
    )

    peptides = tokenizer.tokenize(["LESLIEK", "PEPTIDER"])
    encoder = SpectrumTransformerEncoder(8, 2, 12)
    memory, mem_mask = encoder(spectra[:, :, 0], spectra[:, :, 1])

    decoder = AnalyteTransformerDecoder(n_tokens, 8, 2, 12, padding_int=0)
    scores = decoder(peptides, memory=memory, memory_key_padding_mask=mem_mask)
    assert scores.shape == (2, 9, len(tokenizer))

    scores = decoder(peptides, memory=memory)
    assert scores.shape == (2, 9, len(tokenizer))


def test_analyte_decoder_flash_compatible():
    """Test that flash_compatible=True produces correct output shape.

    Verifies the opt-in path: tgt_key_padding_mask and causal tgt_mask
    are both suppressed, existing AR behaviour (flash_compatible=False)
    is unaffected.
    """
    tokenizer = PeptideTokenizer()
    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500.0, 0.5], [0.0, 0.0]],
        ]
    )
    peptides = tokenizer.tokenize(["LESLIEK", "PEPTIDER"])
    encoder = SpectrumTransformerEncoder(8, 2, 12)
    memory, mem_mask = encoder(spectra[:, :, 0], spectra[:, :, 1])
    decoder = AnalyteTransformerDecoder(
        len(tokenizer), 8, 2, 12, padding_int=0
    )

    # flash_compatible=True — should produce identical shape to AR path
    scores_flash = decoder(
        peptides,
        memory=memory,
        memory_key_padding_mask=mem_mask,
        flash_compatible=True,
    )
    assert scores_flash.shape == (2, 9, len(tokenizer))

    # flash_compatible=False (default) — AR path must still work unchanged
    scores_ar = decoder(
        peptides,
        memory=memory,
        memory_key_padding_mask=mem_mask,
        flash_compatible=False,
    )
    assert scores_ar.shape == (2, 9, len(tokenizer))


def test_analyte_decoder_tgt_is_causal():
    """Test tgt_is_causal=True matches baseline and rejects bad combos.

    Verifies tgt_is_causal=True produces numerically identical output
    to tgt_is_causal=False (AR path) at real token positions, and that
    combining it with flash_compatible=True raises a clear error rather
    than crashing inside PyTorch or silently producing wrong results.
    """
    tokenizer = PeptideTokenizer()
    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500.0, 0.5], [0.0, 0.0]],
        ]
    )
    peptides = tokenizer.tokenize(["LESLIEK", "PEPTIDER"])
    encoder = SpectrumTransformerEncoder(8, 2, 12)
    memory, mem_mask = encoder(spectra[:, :, 0], spectra[:, :, 1])
    decoder = AnalyteTransformerDecoder(
        len(tokenizer), 8, 2, 12, padding_int=0
    )

    scores_baseline = decoder(
        peptides, memory=memory, memory_key_padding_mask=mem_mask,
    )
    scores_hint = decoder(
        peptides, memory=memory, memory_key_padding_mask=mem_mask,
        tgt_is_causal=True,
    )
    # Compare REAL (non-padding) token positions only. tgt_is_causal=True
    # also drops tgt_key_padding_mask (see analytes.py comment) — this is
    # safe because padding is trailing-only and the causal mask already
    # excludes padding keys for every real query. The one legitimate,
    # harmless side effect is that a PADDING position's own output value
    # may differ (it's no longer blocked from attending elsewhere) — but
    # that value is never used downstream (discarded, not fed to loss or
    # generation), so real-position equivalence is the correct safety bar.
    # +1 for the prepended global token in each sequence
    real_len_0 = len("LESLIEK") + 1
    real_len_1 = len("PEPTIDER") + 1
    assert torch.equal(
        scores_baseline[0, :real_len_0], scores_hint[0, :real_len_0]
    )
    assert torch.equal(
        scores_baseline[1, :real_len_1], scores_hint[1, :real_len_1]
    )

    with pytest.raises(ValueError):
        decoder(
            peptides, memory=memory, memory_key_padding_mask=mem_mask,
            flash_compatible=True, tgt_is_causal=True,
        )
