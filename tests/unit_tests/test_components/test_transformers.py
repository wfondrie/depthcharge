"""Test the transformer components"""
import torch
from depthcharge.masses import PeptideMass
from depthcharge.components.transformers import (
    SpectrumEncoder,
    PeptideEncoder,
    PeptideDecoder,
)


def test_spectrum_encoder():
    """Test that a spectrum encoder will run."""
    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500, 0.5], [0, 0]],
        ]
    )

    model = SpectrumEncoder(8, 1, 12, dim_intensity=None)
    emb, mask = model(spectra)
    assert emb.shape == (2, 4, 8)
    assert mask.sum() == 1

    model = SpectrumEncoder(8, 1, 12, dim_intensity=4)
    emb, mask = model(spectra)
    assert emb.shape == (2, 4, 8)
    assert mask.sum() == 1


def test_peptide_encoder():
    """Test that a peptide encoder will run"""
    peptides = ["LESLIEK", "PEPTIDER", "EDITPEPR"]
    charges = torch.tensor([2, 3, 3])

    model = PeptideEncoder(8, 1, 12, max_charge=3)
    emb, mask = model(peptides, charges)

    vocab_size = len(PeptideMass("canonical").masses.keys()) + 1
    assert model.vocab_size == vocab_size
    assert emb.shape == (3, 10, 8)
    assert mask.sum() == 1

    sums = emb.sum(dim=1)
    assert (
        sums[
            1,
            :,
        ]
        != sums[2, :]
    ).all()


def test_peptide_decoder():
    """Test that a peptide decoder will run"""
    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500, 0.5], [0, 0]],
        ]
    )

    peptides = ["LESLIEK", "PEPTIDER"]
    precursors = torch.tensor([[100.0, 2], [200.0, 3]])

    encoder = SpectrumEncoder(8, 1, 12)
    memory, mem_mask = encoder(spectra)

    decoder = PeptideDecoder(8, 1, 12, max_charge=3)
    scores, tokens = decoder(peptides, precursors, memory, mem_mask)
    assert scores.shape == (2, 10, decoder.vocab_size + 1)
    assert tokens.shape == (2, 9)
