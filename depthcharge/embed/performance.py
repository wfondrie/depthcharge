"""Functions to assess model performance"""
import logging

import torch
import numpy as np
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def predicted(loaders, trainer, model, test=False):
    """Assess the performance of a model on a dataset.

    Currently the similarity measure is the generalized scalar product.

    Parameters
    ----------
    loaders : PairedSpectrumDataModule
        The DataLoaders.
    trainer : pytorch_lightning.Trainer
        The Trainer wrapping the model.
    test : bool, optional
        Evaluste the test set? Otherwise the validation set is evaluated.

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
    if test:
        loader = loaders.test_dataloader()
    else:
        loader = loaders.val_dataloader()

    pred = trainer.predict(model, loader)
    pred, similarity = list(zip(*pred))
    pred = torch.cat(pred).detach().cpu().numpy()
    similarity = torch.cat(similarity).detach().cpu().numpy()
    mse = ((similarity - pred) ** 2).mean()
    return pred, similarity, mse


def binned(loaders, bin_width=0.02, offset=0.0, test=False):
    """Compute the scalar product for the discretized spectra.

    Parameters
    ----------
    loaders : PairedSpectrumDataModule
        The DataLoaders.
    bin_width : float
        The m/z bin size for discretization.
    offset : float
        The m/z offset for discretization.
    test : bool, optional
        Evaluste the test set? Otherwise the validation set is evaluated.

    Returns
    -------
    predicted : np.array
        The predicted similarity between the pairs of mass spectra
    similarity : np.array
        The true similarity between pairs of mass spectra.
    mse : float
        The mean squared error.
    """
    LOGGER.info("Computing scalar products with binned spectra...")
    if test:
        loader = loaders.test_dataloader()
    else:
        loader = loaders.val_dataloader()

    pred = []
    similarity = []
    for batch in tqdm(loader):
        for spx, spy, gsp in zip(*batch):
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
