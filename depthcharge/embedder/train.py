"""Train a new SpectrumTransformer"""
import time
import shutil
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from tqdm import tqdm

from . import splits
from .. import utils
from .data import SpectrumDataset
from .model import SpectrumTransformer

LOGGER = logging.getLogger(__name__)


def tmp_data(spectrum_files, tmp_dir):
    """Copy data to a temporary directory.

    This function exists because we mostly work off of a network drive,
    leading to slow I/O speeds. Instead, it is better to create a copy of
    all of the data on the host machine using it's temporary directories.

    Parameters
    ----------
    spectrum_files : str or Path object or list of either
        The mass spectra, either as mzML files or the previously cached
        versions. If using the cached version, you can supply either the
        spectrum file :code:`*.npy` file or the index file :code:`*.index.npy`
        and this function will find the other automatically. You can also
        provide both.

    Returns
    -------
    list of Path objects
        The paths to the spectrum data files.
    """
    out_files = []
    for in_file in tqdm(utils.listify(spectrum_files), unit="files"):
        in_file = Path(in_file)
        out_file = Path(tmp_dir, in_file.name)
        out_files.append(out_file)
        if not out_file.exists():
            shutil.copyfile(in_file, out_file)

        if out_file.suffix == ".npy":
            if out_file.suffixes[-2:] == [".index", ".npy"]:
                sibling_in_file = in_file.with_suffix("").with_suffix(".npy")
            else:
                sibling_in_file = in_file.with_suffix(".index.npy")

            sibling_out_file = Path(tmp_dir, sibling_in_file.name)
            if not sibling_out_file.exists():
                shutil.copyfile(sibling_in_file, sibling_out_file)

    return out_files


def train(
    embed_dim=32,
    training_files=None,
    validation_files=None,
    model_checkpoint=None,
    model_kwargs=None,
    dataset_kwargs=None,
    use_tmp=True,
):
    """Train a SpectrumTransformer

    Train a new SpectrumTransformer model

    Parameters
    ----------
    embed_dim : int
        The embedding dimension.
    training_files : str or Path or list of either
        One or more files containing mass spectra to use for training. These
        should either be in mzML format or be the cached spectrum files.
    validation_fiels : str or Path or list of either
        One or more files containing mass spectra to use for validation. These
        should either be in mzML format or be the cached spectrum files.
    model_checkpoint : str or Path,
        The path to previously saved model weights. If provided, the model
        training will start from here.
    model_kwargs : Dict
        Keyword arguments to pass to the SpectrumTransformer constructor.
    dataset_kwargs : Dict
        Keyword arguments to pass to the SpectrumDataset constructor.
    use_temp : bool
        Copy data to a temporary directory for training. This can be useful
        if the data is normally stored on a network drive, becuase training
        will be much faster if it is local.
    """
    start = time.time()
    if training_files is None and validation_files is None:
        training_files, validation_files, _ = splits.get_splits()
        if "cache_dir" in dataset_kwargs.keys():
            cache = dataset_kwargs["cache_dir"]
            training_files = [Path(cache, f) for f in training_files]
            validation_files = [Path(cache, f) for f in validation_files]

    elif training_files is None and validation_files is not None:
        raise ValueError(
            "'training_files' must be provided if 'validation_files' are "
            "provided"
        )

    with TemporaryDirectory() as tmp:
        if use_tmp:
            training_files = tmp_data(training_files, tmp)
            validation_files = tmp_data(validation_files, tmp)

        train_set = SpectrumDataset(training_files, **dataset_kwargs)
        val_set = SpectrumDataset(validation_files, **dataset_kwargs)
        model = SpectrumTransformer(embed_dim, **model_kwargs)

        LOGGER.info("%i training set mass spectra", len(train_set.spectra))
        LOGGER.info("%i validation set mass spectra", len(val_set.spectra))
        LOGGER.info("Datasets loaded in %.2f min", (time.time() - start) / 60)

        LOGGER.info(
            "Training SpectrumTransformer with %i parameters...",
            model.n_parameters,
        )

        if model_checkpoint is not None:
            model.load(model_checkpoint)

        LOGGER.info("=== Training ===")
        start = time.time()
        model = model.fit(train_set, val_set)

    LOGGER.info("=== Done! ===")
    LOGGER.info("Training completed in %.2f min.", (time.time() - start) / 60)
