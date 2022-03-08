"""Verify that a de novo sequencing model will run."""
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pyteomics.mass import calculate_mass
from depthcharge.denovo import Spec2Pep, DeNovoDataModule
from depthcharge.data import AnnotatedSpectrumIndex


def test_denovo_model(tmp_path):
    """Test de novo model overfitting"""
    pl.seed_everything(42)
    peptides = ["LESLIEK", "PEPTIDEK"]
    mgf_file = tmp_path / "fake.mgf"
    _create_mgf(peptides, mgf_file)

    index = AnnotatedSpectrumIndex(tmp_path / "test.hdf5", mgf_file)
    loaders = DeNovoDataModule(index, num_workers=0, batch_size=2)
    loaders.setup()

    # Create the model
    model = Spec2Pep(16, n_layers=3, n_log=20, dim_intensity=4)
    trainer = pl.Trainer(
        logger=None, max_epochs=300, checkpoint_callback=False
    )

    # Fit the model
    trainer.fit(model, loaders.train_dataloader())

    # Predict
    pred = trainer.predict(model, loaders.train_dataloader())
    assert pred[0][0][0][0] == "$LESLLEK"


def _create_mgf_entry(peptide, charge=2):
    """Create a MassIVE-KB style MGF entry for a single PSM.

    Parameters
    ----------
    peptide : str
        A peptide sequence.
    charge : int, optional
        The peptide charge state.

    Returns
    -------
    str
        The PSM entry in an MGF file format.
    """
    mz = calculate_mass(peptide, charge=int(charge))
    frags = []
    for idx in range(len(peptide)):
        for zstate in range(1, charge):
            b_pep = peptide[: idx + 1]
            frags.append(
                str(calculate_mass(b_pep, charge=zstate, ion_type="b"))
            )

            y_pep = peptide[idx:]
            frags.append(
                str(calculate_mass(y_pep, charge=zstate, ion_type="y"))
            )

    frag_string = " 1\n".join(frags) + " 1"

    mgf = [
        "BEGIN IONS",
        f"SEQ={peptide}",
        f"PEPMASS={mz}",
        f"CHARGE={charge}+",
        f"{frag_string}",
        "END IONS",
    ]
    return "\n".join(mgf)


def _create_mgf(peptides, mgf_file, random_state=42):
    """Create a fake MGF file from one or more peptides.

    Parameters
    ----------
    peptides : str or list of str
        The peptides for which to create spectra.
    mgf_file : str or Path
        The MGF file to create.
    random_state : int or numpy.random.Generator, optional
        The random seed. The charge states are chosen to be
        2 or 3 randomly.

    Returns
    -------
    """
    rng = np.random.default_rng(random_state)
    mgf_file = Path(mgf_file)
    entries = [_create_mgf_entry(p, rng.choice([2, 3])) for p in peptides]
    with mgf_file.open("w+") as mgf_ref:
        mgf_ref.write("\n".join(entries))

    return mgf_file
