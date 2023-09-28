"""Tranformer models to handle mass spectra."""
from collections.abc import Callable

import torch

from ..encoders import PeakEncoder


class SpectrumTransformerEncoder(torch.nn.Module):
    """A Transformer encoder for input mass spectra.

    Parameters
    ----------
    d_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    nhead : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``nhead``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    peak_encoder : PeakEncoder or bool, optional
        Sinusoidal encodings m/z and intensityvalues of each peak.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoder | Callable | bool = True,
    ) -> None:
        """Initialize a SpectrumEncoder."""
        super().__init__()
        self._d_model = d_model
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward
        self._n_layers = n_layers
        self._dropout = dropout

        if callable(peak_encoder):
            self.peak_encoder = peak_encoder
        elif peak_encoder:
            self.peak_encoder = PeakEncoder(d_model)
        else:
            self.peak_encoder = torch.nn.Linear(2, d_model)

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    @property
    def d_model(self) -> int:
        """The latent dimensionality of the model."""
        return self._d_model

    @property
    def nhead(self) -> int:
        """The number of attention heads."""
        return self._nhead

    @property
    def dim_feedforward(self) -> int:
        """The dimensionality of the Transformer feedforward layers."""
        return self._dim_feedforward

    @property
    def n_layers(self) -> int:
        """The number of Transformer layers."""
        return self._n_layers

    @property
    def dropout(self) -> float:
        """The dropout for the transformer layers."""
        return self._dropout

    def forward(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a batch of mass spectra.

        Parameters
        ----------
        mz_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded m/z dimension for a batch of mass spectra.
        intensity_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded intensity dimension for a batch of mass spctra.
        **kwargs : dict
            Additional fields provided by the data loader. These may be
            used by overwriting the `precursor_hook()` method in a subclass.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, d_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        spectra = torch.stack([mz_array, intensity_array], dim=2)
        n_batch = spectra.shape[0]
        zeros = ~spectra.sum(dim=2).bool()
        mask = torch.cat(
            [torch.tensor([[False]] * n_batch).type_as(zeros), zeros], dim=1
        )
        peaks = self.peak_encoder(spectra)

        # Add the precursor information:
        latent_spectra = self.precursor_hook(
            mz_array=mz_array,
            intensity_array=intensity_array,
            **kwargs,
        )

        peaks = torch.cat([latent_spectra[:, None, :], peaks], dim=1)
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask

    def precursor_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """Define how additional information in the batch may be used.

        Overwrite this method to define custom functionality dependent on
        information in the batch. Examples would be to incorporate any
        combination of the mass, charge, retention time, or
        ion mobility of a precursor ion.

        The representation returned by this method is preprended to the
        peak representations that are fed into the Transformer encoder and
        ultimately contribute to the spectrum representation that is the
        first element of the sequence in the model output.

        By default, this method returns a tensor of zeros.

        Parameters
        ----------
        mz_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded m/z dimension for a batch of mass spectra.
        intensity_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded intensity dimension for a batch of mass spctra.
        **kwargs : dict
            The additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The precursor representations.
        """
        return torch.zeros((mz_array.shape[0], self.d_model)).type_as(mz_array)

    @property
    def device(self) -> torch.device:
        """The current device for the model."""
        return next(self.parameters()).device
