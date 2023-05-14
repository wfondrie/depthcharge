"""Similarity measures between mass spectra."""
import numba as nb
import numpy as np


@nb.njit
def gsp(spectrum_x, spectrum_y, tol):
    """Compute the dot product between two spectra.

    This code was adapted from Wout:
    https://github.com/bittremieux/GLEAMS/blob/master/gleams/feature/spectrum.py#L154


    Parameters
    ----------
    spectrum_x : np.ndarray
    spectrum_y : np.ndarray
        The spectra to compare. Each row should be a peak, with the first
        column indicating the m/z and the second column indicating the
        intensities.
    tol : float
        The fragment m/z tolerance used to match peaks in both spectra with
        each other.

    Returns
    -------
    float
        The dot product between both spectra.
    """
    # Find the matching peaks between both spectra.
    peak_match_scores, peak_match_idx = [], []
    peak_other_i = 0
    mz_other, intensity_other = spectrum_y.T
    for peak_i, (peak_mz, peak_intensity) in enumerate(spectrum_x):
        # Advance while there is an excessive mass difference.
        while (
            peak_other_i < len(mz_other) - 1
            and peak_mz - tol > mz_other[peak_other_i]
        ):
            peak_other_i += 1
        # Match the peaks within the fragment mass window if possible.
        peak_other_window_i = peak_other_i
        while (
            peak_other_window_i < len(mz_other)
            and abs(peak_mz - (mz_other[peak_other_window_i])) <= tol
        ):
            peak_match_scores.append(
                peak_intensity * intensity_other[peak_other_window_i]
            )
            peak_match_idx.append((peak_i, peak_other_window_i))
            peak_other_window_i += 1

    score = 0
    if len(peak_match_scores) > 0:
        # Use the most prominent peak matches to compute the score (sort in
        # descending order).
        peak_match_scores_arr = np.asarray(peak_match_scores)
        peak_match_order = np.argsort(peak_match_scores_arr)[::-1]
        peak_match_scores_arr = peak_match_scores_arr[peak_match_order]
        peak_match_idx_arr = np.asarray(peak_match_idx)[peak_match_order]
        peaks_used, peaks_used_other = set(), set()
        for peak_match_score, peak_i, peak_other_i in zip(
            peak_match_scores_arr,
            peak_match_idx_arr[:, 0],
            peak_match_idx_arr[:, 1],
        ):
            if (
                peak_i not in peaks_used
                and peak_other_i not in peaks_used_other
            ):
                score += peak_match_score
                # Make sure these peaks are not used anymore.
                peaks_used.add(peak_i)
                peaks_used_other.add(peak_other_i)

    return score
