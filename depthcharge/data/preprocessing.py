"""Preprocessing functions for mass spectra.

These functions can be used with any depthcharge dataset or
data module.


"""
from typing import Tuple

import torch


def sqrt_and_norm(
    mz_array: torch.Tensor,
    int_array: torch.Tensor,
    precursor_mz: float,
    precursor_charge: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Square root the intensities and scale to unit norm.

    Parameters
    ----------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    precursor_mz : float
        The precursor m/z.
    precursor_charge : int
        The precursor charge.

    Returns
    -------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    """
    print(int_array)
    int_array = torch.sqrt(int_array)
    int_array /= torch.linalg.norm(int_array)
    return mz_array, int_array


def remove_precursor_peak(
    mz_array: torch.Tensor,
    int_array: torch.Tensor,
    precursor_mz: float,
    tol: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:

    """Square root the intensities and scale to unit norm.

    Parameters
    ----------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    precursor_mz : float
        The precursor m/z.
    precursor_charge : int
        The precursor charge.
    tol : float, optional
        The tolerance to remove the precursor peak in m/z.

    Returns
    -------
    mz_array : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    int_array : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    """
    bounds = (precursor_mz - tol, precursor_mz + tol)
    outside = (mz_array >= bounds[0]) | (mz_array <= bounds[1])
    return mz_array[outside], int_array[outside]
