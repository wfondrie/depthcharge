"""Simple encoders for input into Transformers and the like."""
import torch
import einops
import numpy as np


class MassEncoder(torch.nn.Module):
    """Encode mass values using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use.
    max_wavelength : float
        The maximum wavelength to use.
    """

    def __init__(self, dim_model, min_wavelength=0.001, max_wavelength=10000):
        """Initialize the MassEncoder"""
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin

        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1
            scale = max_wavelength / (2 * np.pi)

        sin_term = base * scale ** (
            torch.arange(0, n_sin).float() / (n_sin - 1)
        )
        cos_term = base * scale ** (
            torch.arange(0, n_cos).float() / (n_cos - 1)
        )

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (n_masses)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (n_masses, dim_model)
            The encoded features for the mass spectra.
        """
        sin_mz = torch.sin(X / self.sin_term)
        cos_mz = torch.cos(X / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)


class PeakEncoder(MassEncoder):
    """Encode m/z values in a mass spectrum using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    dim_intensity : int, optional
        The number of features to use for intensity. The remaining features
        will be used to encode the m/z values.
    min_wavelength : float, optional
        The minimum wavelength to use.
    max_wavelength : float, optional
        The maximum wavelength to use.
    """

    def __init__(
        self,
        dim_model,
        dim_intensity=None,
        min_wavelength=0.001,
        max_wavelength=10000,
    ):
        """Initialize the MzEncoder"""
        self.dim_intensity = dim_intensity
        self.dim_mz = dim_model
        if self.dim_intensity is not None:
            self.dim_mz -= self.dim_intensity

        super().__init__(
            dim_model=self.dim_mz,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

        if self.dim_intensity is None:
            self.int_encoder = torch.nn.Linear(1, dim_model, bias=False)
        else:
            self.int_encoder = MassEncoder(
                dim_model=dim_intensity,
                min_wavelength=0,
                max_wavelength=1,
            )

    def forward(self, X):
        """Encode m/z values and intensities.

        Note that we expect intensities to fall within the interval [0, 1].

        Parameters
        ----------
        X : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        torch.Tensor of shape (n_spectr, n_peaks, dim_model)
            The encoded features for the mass spectra.
        """
        m_over_z = X[:, :, [0]]
        encoded = super().forward(m_over_z)
        intensity = self.int_encoder(X[:, :, [1]])
        if self.dim_intensity is None:
            return encoded + intensity

        return torch.cat([encoded, intensity], dim=2)


class PositionalEncoder(torch.nn.Module):
    """The positional encoder for sequences.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    """

    def __init__(self, dim_model, max_wavelength=10000):
        """Initialize the MzEncoder"""
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin
        scale = max_wavelength / (2 * np.pi)

        sin_term = scale ** (torch.arange(0, n_sin).float() / (n_sin - 1))
        cos_term = scale ** (torch.arange(0, n_cos).float() / (n_cos - 1))
        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode positions in a sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_sequence, n_features)
            The first dimension should be the batch size (i.e. each is one
            peptide) and the second dimension should be the sequence (i.e.
            each should be an amino acid representation).

        Returns
        -------
        torch.Tensor of shape (batch_size, n_sequence, n_features)
            The encoded features for the mass spectra.
        """
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X
