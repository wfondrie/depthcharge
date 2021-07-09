"""Functions to assess model performance"""
import logging

import torch
import numpy as np
from tqdm import tqdm

from .model import prepare_batch

LOGGER = logging.getLogger(__name__)


def predicted(model, dataset):
    """Assess the performance of a model on a dataset.

    Currently the similarity measure is the generalized scalar product.

    Parameters
    ----------
    dataset : SpectrumDataset
        The dataset to evaluate the model with.
    model : SpectrumTransformer
        The model to evaluate.

    Returns
    -------
    predicted : np.array
        The predicted similarity between the pairs of mass spectra
    similarity : np.array
        The true similarity between pairs of mass spectra.
    mse : float
        The mean squared error.
    """
    LOGGER.info("Predicting with model...")
    model.eval()
    model.to(model._device)
    dataset.eval()
    loader = torch.utils.data.DataLoader(
        dataset, model.eval_batch_size, collate_fn=prepare_batch
    )
    pred = []
    similarity = []
    with torch.no_grad():
        for spx, spy, gsp in tqdm(loader):
            similarity.append(gsp.flatten().to("cpu"))
            pred.append(model.predict(spx, spy)[-1].flatten().to("cpu"))

    pred = torch.cat(pred).detach().numpy()
    similarity = torch.cat(similarity).detach().numpy()
    mse = ((similarity - pred) ** 2).mean()
    return pred, similarity, mse


def binned(dataset, bin_width=0.02, offset=0.0):
    """Compute the scalar product for the discretized spectra"""
    LOGGER.info("Computing scalar products with binned spectra...")
    dataset.eval()
    pred = []
    similarity = []
    for spx, spy, gsp in tqdm(dataset):
        spx = _vectorize(spx.cpu().detach().numpy(), bin_width, offset)
        spy = _vectorize(spy.cpu().detach().numpy(), bin_width, offset)
        diff = len(spx) - len(spy)
        if diff > 0:
            spy = np.pad(spy, (0, diff))
        elif diff < 0:
            spx = np.pad(spx, (0, -diff))

        spx = spx / np.linalg.norm(spx)
        spy = spy / np.linalg.norm(spy)
        pred.append(np.dot(spx, spy))
        similarity.append(gsp)

    similarity = np.array(similarity)
    pred = np.array(pred)
    mse = ((similarity - pred) ** 2).mean()
    return pred, similarity, mse


def _vectorize(spectrum, bin_width, offset):
    """Vectorize a mass spectrum

    Parameters
    ----------
    spectrum : np.array
        A 2-dimensional numpy array representing a mass spectrum. The first
        column should be the m/z values and the second column should be the
        intensity values.
    """
    if not spectrum.size:
        return np.array([0.00001])

    mz_bins, indices = np.unique(
        np.floor((spectrum[:, 0] / bin_width) + 1 - offset).astype(int),
        return_index=True,
    )

    mz_bins = np.array(mz_bins)
    intensities = np.array([spectrum[i, 1].max() for i in indices])
    vec = np.zeros(mz_bins.max() + 1)
    vec[mz_bins] = intensities
    return np.sqrt(vec)
