"""Test the spectrum transformers."""
import torch

from depthcharge.encoders import PeakEncoder
from depthcharge.transformers import SpectrumTransformerEncoder


def test_spectrum_encoder():
    """Test that a spectrum encoder will run."""
    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500, 0.5], [0, 0]],
        ]
    )

    model = SpectrumTransformerEncoder(8, 1, 12)
    emb, mask = model(spectra)
    assert emb.shape == (2, 4, 8)
    assert mask.sum() == 1

    model = SpectrumTransformerEncoder(8, 1, 12, peak_encoder=PeakEncoder(8))
    emb, mask = model(spectra)
    assert emb.shape == (2, 4, 8)
    assert mask.sum() == 1

    model = SpectrumTransformerEncoder(8, 1, 12, peak_encoder=False)
    emb, mask = model(spectra)
    assert emb.shape == (2, 4, 8)
    assert mask.sum() == 1
