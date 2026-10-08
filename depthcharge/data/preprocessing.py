"""Preprocessing functions for mass spectra.

Preprocessing functions are applied to each mass spectrum during parsing,
using the `preprocessing_fn` parameter of `spectra_to_df()`,
`spectra_to_parquet()`, and `spectra_to_stream()`. For datasets, pass
`preprocessing_fn` in `parse_kwargs` (`SpectrumDataset` and
`AnnotatedSpectrumDataset`) or as a keyword argument
(`StreamingSpectrumDataset`). To apply preprocessing steps sequentially, pass
a list of functions.

The following functions wrap the
[spectrum_utils](https://spectrum-utils.readthedocs.io) `MsmsSpectrum`
methods of the same name. Calling one with that method's arguments returns a
preprocessing function:

- `filter_intensity()`
- `remove_precursor_peak()`
- `round()`
- `scale_intensity()`
- `set_mz_range()`

Additionally, `scale_to_unit_norm()` is a preprocessing function itself.

We can also define custom preprocessing functions. All preprocessing functions
must accept a `MassSpectrum` as their only argument and return a
`MassSpectrum`. If a `MassSpectrum` is invalid, the function should raise a
`ValueError` and the spectrum will be skipped.

### Examples

Remove the peaks around the precursor m/z, then square root transform
intensities and scale to unit norm:
```python
from depthcharge.data import SpectrumDataset, preprocessing

SpectrumDataset(
    ...,
    parse_kwargs={
        "preprocessing_fn": [
            preprocessing.remove_precursor_peak(0.1, "Da"),
            preprocessing.scale_intensity("root"),
            preprocessing.scale_to_unit_norm,
        ],
    },
)
```

Apply a custom function:
```python
import numpy as np

def log_intensity(spectrum: MassSpectrum) -> MassSpectrum:
    spectrum.intensity = np.log1p(spectrum.intensity)
    return spectrum

SpectrumDataset(
    ...,
    parse_kwargs={"preprocessing_fn": log_intensity},
)
```

"""

from collections.abc import Callable
from functools import wraps

import numpy as np

from ..primitives import MassSpectrum


def scale_to_unit_norm(spectrum: MassSpectrum) -> MassSpectrum:
    """Scale intensities to unit norm."""
    spectrum.intensity = (
        spectrum.intensity / np.sqrt(spectrum.intensity**2).sum()
    )
    return spectrum


def _spectrum_utils_fn(func: str) -> Callable:
    """Wrap spectrum_utils.spectrum.MsmsSpectrum preprocessing methods."""

    @wraps(getattr(MassSpectrum, func))
    def wrapper(
        *args: tuple,
        **kwargs: dict,
    ) -> Callable:
        """Wrapper for spectrum_utils MsmsSpectrum methods.

        Parameters
        ----------
        func: Callable
            The preprocessing function. This should exactly match a
            MsmsSpectrum method.
        *args: tuple
            Arguments that are passed to the MsmsSpectrum method.
        **kwargs : dict
            Keyword arguments that are passed to the MsmsSpectrum method.

        Returns
        -------
        Callable
            A valid depthcharge preprocessing function.

        """

        @wraps(wrapper)
        def preprocess(spec: MassSpectrum) -> MassSpectrum:
            """The wrapped preprocessing function.

            Parameters
            ----------
            spec : MassSpectrum
                The mass spectrum to preprocess

            Returns
            -------
            MassSpectrum
                The processed mass spectrum.

            """
            # Call the spectrum_utils method:
            getattr(spec, func)(*args, **kwargs)
            return spec

        return preprocess

    return wrapper


def _add_spectrum_utils_methods() -> None:
    """Update this module with the spectrum_utils MsmsSpectrum methods."""
    sus_methods = [
        "filter_intensity",
        "remove_precursor_peak",
        "round",
        "scale_intensity",
        "set_mz_range",
    ]

    for method in sus_methods:
        globals()[method] = _spectrum_utils_fn(method)


_add_spectrum_utils_methods()
