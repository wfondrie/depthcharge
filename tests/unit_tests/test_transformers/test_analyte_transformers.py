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
