"""Preprocessing functions for mass spectra.

These functions can be used with datasets (`SpectrumDataset` and
`AnnotatedSpectrumDataset`) or our `SpectrumDataModule`.

One or more preprocessing function can be applied to each mass spectrum during
batch loading using the `preprocessing_fn` parameter of these classes. To apply
preprocessing steps sequentially, merely pass a list of functions to this
argument.

We can also define custom preprocessing functions. All preprocessing functions
must accept 4 keyword arguments:
    - `mz_array` : torch.Tensor of shape (n_peaks,)
        The m/z values of the peaks in the spectrum.
    - `int_array` : torch.Tensor of shape (n_peaks,)
        The intensity values of the peaks in the spectrum.
    - `precursor_mz` : float
        The precursor m/z.
    - `precursor_charge` : int
        The precursor charge.

Preprocessing functions always return the processed `mz_array` and `int_array`.


### Examples

Remove the peaks around the precursor m/z then square root transform
intensities and scale to unit norm:
```Python
SpectrumDataset(
    ...,
    preprocessing_fn=[
        preprocessing.remove_precursor_preak,
        preprocessing.sqrt_and_norm,
    ],
)
```

Apply a custom function:
```
def my_func(mz_array, int_array, precursor_mz, precursor_charge):
    return mz_array, torch.log(int_array)

SpectrumDataset(
    ...,
    preprocessing_fn=my_func,
)
```

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
    int_array = torch.sqrt(int_array)
    int_array /= torch.linalg.norm(int_array)
    return mz_array, int_array


def remove_precursor_peak(
    mz_array: torch.Tensor,
    int_array: torch.Tensor,
    precursor_mz: float,
    tol: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove peaks near the precursor m/z.

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
