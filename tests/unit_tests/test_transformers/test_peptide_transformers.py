"""Test the peptide transformers."""
import torch

from depthcharge.tokenizers import PeptideTokenizer
from depthcharge.transformers import (
    PeptideTransformerDecoder,
    PeptideTransformerEncoder,
    SpectrumTransformerEncoder,
)


def test_peptide_encoder():
    """Test that a peptide encoder will run."""
    tokenizer = PeptideTokenizer()
    peptides = tokenizer.tokenize(["LESLIEK", "PEPTIDER", "EDITHYKK"])
    charges = torch.tensor([2, 3, 3])

    model = PeptideTransformerEncoder(tokenizer, 8, 1, 12, max_charge=3)
    emb, mask = model(peptides, charges)

    # Axis 1 should be 1 longer than the longest peptide.
    assert emb.shape == (3, 9, 8)
    assert mask.sum() == 1

    res = emb.sum(dim=1)
    assert (res[1, :] != res[2, :]).all()


def test_peptide_decoder():
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
    precursors = torch.tensor([[100.0, 2], [200.0, 3]])

    encoder = SpectrumTransformerEncoder(8, 1, 12)
    memory, mem_mask = encoder(spectra)

    decoder = PeptideTransformerDecoder(n_tokens, 8, 1, 12, max_charge=3)
    scores = decoder(peptides, precursors, memory, mem_mask)
    assert scores.shape == (2, 9, len(tokenizer))
