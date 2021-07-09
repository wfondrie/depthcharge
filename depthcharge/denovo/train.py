"""Train a new Spec2Pep model"""
import time
import shutil
import logging
import tempfile
from pathlib import Path

from tqdm import tqdm

from .. import utils
from .model import Spec2Pep
from ..embedder.model import SpectrumTransformer
from ..data import AnnotatedSpectrumDataset

LOGGER = logging.getLogger(__name__)


def tmp_data(spectrum_files):
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
    tmp_dir = tempfile.gettempdir()
    out_files = []
    LOGGER.info("Copying files to temporary directory...")
    for in_file in tqdm(utils.listify(spectrum_files), unit="files"):
        in_file = Path(in_file)
        out_file = Path(tmp_dir, in_file.name)
        out_files.append(out_file)
        if not out_file.exists():
            shutil.copyfile(in_file, out_file)

        if out_file.suffix == ".npy":
            root = str(in_file).replace(".npy", "")
            root = root.replace(".precursors", "")
            root = root.replace(".peptides", "")

            suffixes = [".index", ".precursors", ".peptides", ""]
            for suffix in suffixes:
                in_file = Path(root + suffix + ".npy")
                out_file = Path(tmp_dir, in_file.name)
                if not out_file.exists():
                    shutil.copyfile(in_file, out_file)

    return out_files


def train(
    embed_dim=32,
    training_files=None,
    validation_files=None,
    encoder_checkpoint=None,
    model_checkpoint=None,
    encoder_kwargs=None,
    decoder_kwargs=None,
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
    if encoder_kwargs is None:
        encoder_kwargs = {}

    if decoder_kwargs is None:
        decoder_kwargs = {}

    if dataset_kwargs is None:
        dataset_kwargs = {}

    if use_tmp:
        training_files = tmp_data(training_files)
        validation_files = tmp_data(validation_files)

    LOGGER.info("Creating AnnotatedSpectrumDatasets...")
    train_set = AnnotatedSpectrumDataset(training_files, **dataset_kwargs)
    val_set = AnnotatedSpectrumDataset(validation_files, **dataset_kwargs)
    encoder = SpectrumTransformer(embed_dim, **encoder_kwargs)
    model = Spec2Pep(encoder, **decoder_kwargs)

    LOGGER.info("%i training set mass spectra", train_set.n_spectra)
    LOGGER.info("%i validation set mass spectra", val_set.n_spectra)
    LOGGER.info("Datasets loaded in %.2f min", (time.time() - start) / 60)
    LOGGER.info(
        "Training Spec2Pep model with %i parameters...",
        model.n_parameters,
    )

    if model_checkpoint is not None:
        model.load(model_checkpoint)

    LOGGER.info("=== Training ===")
    start = time.time()
    model.fit(train_set, val_set)

    LOGGER.info("=== Done! ===")
    LOGGER.info("Training completed in %.2f min.", (time.time() - start) / 60)
    return model, val_set
