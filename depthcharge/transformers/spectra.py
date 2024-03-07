"""Tranformer models to handle mass spectra."""

from collections.abc import Callable

import torch

from ..encoders import PeakEncoder
from ..mixins import ModelMixin, TransformerMixin


class SpectrumTransformerEncoder(
    torch.nn.Module, ModelMixin, TransformerMixin
):
    """A Transformer encoder for input mass spectra.

    Use this PyTorch module to embed mass spectra. By default, nothing
    other than the m/z and intensity arrays for each mass spectrum are
    considered. However, arbitrary information can be integrated into the
    spectrum representation by subclassing this class and overwriting the
    `global_token_hook()` method.

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
        The function to encode the (m/z, intensity) tuples of each mass
        spectrum. `True` uses the default sinusoidal encoding and `False`
        instead performs a 1 to `d_model` learned linear projection.

    Attributes
    ----------
    d_model : int
    nhead : int
    dim_feedforward : int
    n_layers : int
    dropout : float
    peak_encoder : torch.nn.Module or Callable
        The function to encode the (m/z, intensity) tuples of each mass
        spectrum.
    transformer_encoder : torch.nn.TransformerEncoder
        The Transformer encoder layers.

    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0.0,
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
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
            dropout=self.dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=self.n_layers,
        )

    def forward(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a batch of mass spectra.

        Parameters
        ----------
        mz_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded m/z dimension for a batch of mass spectra.
        intensity_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded intensity dimension for a batch of mass spctra.
        *args : torch.Tensor
            Additional data. These may be used by overwriting the
            `global_token_hook()` method in a subclass.
        mask : torch.Tensor
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask
            for the sequence.
        **kwargs : dict
            Additional data fields. These may be used by overwriting
            the `global_token_hook()` method in a subclass.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, d_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.

        """
        spectra = torch.stack([mz_array, intensity_array], dim=2)

        # Create the padding mask:
        src_key_padding_mask = spectra.sum(dim=2) == 0
        global_token_mask = torch.tensor([[False]] * spectra.shape[0]).type_as(
            src_key_padding_mask
        )
        src_key_padding_mask = torch.cat(
            [global_token_mask, src_key_padding_mask], dim=1
        )

        # Encode the peaks
        peaks = self.peak_encoder(spectra)

        # Add the precursor information:
        latent_spectra = self.global_token_hook(
            *args,
            mz_array=mz_array,
            intensity_array=intensity_array,
            **kwargs,
        )

        peaks = torch.cat([latent_spectra[:, None, :], peaks], dim=1)
        out = self.transformer_encoder(
            peaks,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        return out, src_key_padding_mask

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
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
        *args : torch.Tensor
            Additional data passed with the batch.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The precursor representations.

        """
        return torch.zeros((mz_array.shape[0], self.d_model)).type_as(mz_array)
