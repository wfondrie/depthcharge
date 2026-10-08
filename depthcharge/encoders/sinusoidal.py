"""Simple encoders for input into Transformers and the like."""

import math
import warnings

import einops
import numpy as np
import torch


class FloatEncoder(torch.nn.Module):
    """Encode floating point values using sine and cosine waves.

    Parameters
    ----------
    d_model : int
        The number of features to output.
    min_wavelength : float, optional
        The minimum wavelength to use.
    max_wavelength : float, optional
        The maximum wavelength to use.
    learnable_wavelengths : bool, optional
        Allow the selected wavelengths to be fine-tuned
        by the model.

    """

    def __init__(
        self,
        d_model: int,
        min_wavelength: float = 0.001,
        max_wavelength: float = 10000,
        learnable_wavelengths: bool = False,
    ) -> None:
        """Initialize the MassEncoder."""
        super().__init__()

        # Error checking:
        if min_wavelength <= 0:
            raise ValueError("'min_wavelength' must be greater than 0.")

        if max_wavelength <= 0:
            raise ValueError("'max_wavelength' must be greater than 0.")

        self.learnable_wavelengths = learnable_wavelengths

        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength
        self._d_model = d_model

        # Dummy buffer to track the model's dtype for output conversion
        self.register_buffer(
            "_dtype_tracker", torch.zeros(1).float(), persistent=False
        )

        sin_term, cos_term = self._compute_wavelength_terms()

        if not self.learnable_wavelengths:
            self.register_buffer("sin_term", sin_term)
            self.register_buffer("cos_term", cos_term)
        else:
            self.sin_term = torch.nn.Parameter(sin_term)
            self.cos_term = torch.nn.Parameter(cos_term)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_float)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_float, d_model)
            The encoded features for the floating point numbers.

        """
        if X.dtype.itemsize < torch.float32.itemsize:
            warnings.warn(
                f"Input tensor has dtype {X.dtype} which is lower precision "
                f"than float32. This may lead to numerical precision issues "
                f"with large input values (e.g. mz arrays). "
                f"It is highly recommended to use float32 inputs.",
            )

        sin_mz = torch.sin(X[:, :, None] / self.sin_term)
        cos_mz = torch.cos(X[:, :, None] / self.cos_term)
        encoded = torch.cat([sin_mz, cos_mz], axis=-1)

        return encoded.to(self._dtype_tracker.dtype)

    def _compute_wavelength_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sin and cos wavelength terms for encoding."""
        d_sin = math.ceil(self._d_model / 2)
        d_cos = self._d_model - d_sin
        base = self._min_wavelength / (2 * np.pi)
        scale = self._max_wavelength / self._min_wavelength
        sin_exp = torch.arange(0, d_sin).float() / (d_sin - 1)
        cos_exp = (torch.arange(d_sin, self._d_model).float() - d_sin) / (
            d_cos - 1
        )
        sin_term = base * (scale**sin_exp)
        cos_term = base * (scale**cos_exp)
        return sin_term, cos_term

    def _apply(self, fn):
        """Override _apply to keep encoding buffers at float32 precision."""
        super()._apply(fn)

        # if not learnable, reconstruct the buffers at float32 precision
        if not self.learnable_wavelengths:
            device = self.sin_term.device
            sin_term, cos_term = self._compute_wavelength_terms()
            self.sin_term = sin_term.to(device)
            self.cos_term = cos_term.to(device)

        return self


class PeakEncoder(torch.nn.Module):
    """Encode mass spectrum.

    Parameters
    ----------
    d_model : int
        The number of features to output.
    min_mz_wavelength : float, optional
        The minimum wavelength to use for m/z.
    max_mz_wavelength : float, optional
        The maximum wavelength to use for m/z.
    min_intensity_wavelength : float, optional
        The minimum wavelength to use for intensity. The default assumes
        intensities between [0, 1].
    max_intensity_wavelength : float, optional
        The maximum wavelength to use for intensity. The default assumes
        intensities between [0, 1].
    learnable_wavelengths : bool, optional
        Allow the selected wavelengths to be fine-tuned
        by the model.

    """

    def __init__(
        self,
        d_model: int,
        min_mz_wavelength: float = 0.001,
        max_mz_wavelength: float = 10000,
        min_intensity_wavelength: float = 1e-6,
        max_intensity_wavelength: float = 1,
        learnable_wavelengths: bool = False,
    ) -> None:
        """Initialize the MzEncoder."""
        super().__init__()
        self.d_model = d_model
        self.learnable_wavelengths = learnable_wavelengths

        self.mz_encoder = FloatEncoder(
            d_model=self.d_model,
            min_wavelength=min_mz_wavelength,
            max_wavelength=max_mz_wavelength,
            learnable_wavelengths=learnable_wavelengths,
        )

        self.int_encoder = FloatEncoder(
            d_model=self.d_model,
            min_wavelength=min_intensity_wavelength,
            max_wavelength=max_intensity_wavelength,
            learnable_wavelengths=learnable_wavelengths,
        )
        self.combiner = torch.nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
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
        torch.Tensor of shape (n_spectra, n_peaks, d_model)
            The encoded features for the mass spectra.

        """
        encoded = torch.cat(
            [
                self.mz_encoder(X[:, :, 0]),
                self.int_encoder(X[:, :, 1]),
            ],
            dim=2,
        )

        return self.combiner(encoded)


class PositionalEncoder(FloatEncoder):
    """The positional encoder for sequences.

    Parameters
    ----------
    d_model : int
        The number of features to output.
    min_wavelength : float, optional
        The shortest wavelength in the geometric progression.
    max_wavelength : float, optional
        The longest wavelength in the geometric progression.

    """

    def __init__(
        self,
        d_model: int,
        min_wavelength: float = 1.0,
        max_wavelength: float = 1e5,
    ) -> None:
        """Initialize the MzEncoder."""
        super().__init__(
            d_model=d_model,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
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
        encoded = torch.cat([sin_pos, cos_pos], axis=2).to(
            self._dtype_tracker.dtype
        )

        # Warn if input dtype doesn't match model dtype
        if X.dtype != encoded.dtype:
            warnings.warn(
                f"Input tensor dtype ({X.dtype}) does not match model dtype "
                f"({encoded.dtype}). The addition will implicitly cast to a "
                f"common dtype, which may cause unexpected behavior.",
            )

        return encoded + X
