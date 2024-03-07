"""Test the spectrum transformers."""

import pytest
import torch

from depthcharge.encoders import PeakEncoder
from depthcharge.transformers import SpectrumTransformerEncoder


@pytest.fixture
def batch():
    """A mass spectrum."""
    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500, 0.5], [0, 0]],
        ]
    )

    batch_dict = {
        "mz_array": spectra[:, :, 0],
        "intensity_array": spectra[:, :, 0],
        "charge": torch.tensor([1.0, 2.0]),
    }

    return batch_dict


def test_spectrum_encoder(batch):
    """Test that a spectrum encoder will run."""
    model = SpectrumTransformerEncoder(8, 2, 12)
    emb, mask = model(**batch)
    assert emb.shape == (2, 4, 8)
    assert mask.sum() == 1

    model = SpectrumTransformerEncoder(8, 2, 12, peak_encoder=PeakEncoder(8))
    emb, mask = model(**batch)
    assert emb.shape == (2, 4, 8)
    assert mask.sum() == 1

    model = SpectrumTransformerEncoder(8, 2, 12, peak_encoder=False)
    emb, mask = model(**batch)
    assert emb.shape == (2, 4, 8)
    assert mask.sum() == 1


def test_global_token_hook(batch):
    """Test that the hook works."""

    class MyEncoder(SpectrumTransformerEncoder):
        """A silly class."""

        def global_token_hook(self, mz_array, intensity_array, **kwargs):
            """A silly hook."""
            return kwargs["charge"].expand(self.d_model, -1).T

    model1 = MyEncoder(8, 2, 12)
    emb1, mask1 = model1(**batch)
    assert emb1.shape == (2, 4, 8)
    assert mask1.sum() == 1

    model2 = SpectrumTransformerEncoder(8, 2, 12)
    emb2, mask2 = model2(**batch)
    assert emb2.shape == (2, 4, 8)
    assert mask2.sum() == 1

    for elem in zip(emb1.flatten(), emb2.flatten()):
        if elem:
            assert elem[0] != elem[1]


def test_properties():
    """Test that the properties work."""
    model = SpectrumTransformerEncoder(
        d_model=256,
        nhead=16,
        dim_feedforward=48,
        n_layers=3,
        dropout=0.1,
    )

    assert model.d_model == 256
    assert model.nhead == 16
    assert model.dim_feedforward == 48
    assert model.n_layers == 3
    assert model.dropout == 0.1
