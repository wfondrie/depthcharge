"""Pytest fixtures."""

from pathlib import Path

import numpy as np
import pytest
from pyteomics.mass import calculate_mass

DATA_DIR = Path(__file__).parent / ".." / "data"


@pytest.fixture
def real_mzml():
    """Get a real mzML file."""
    return DATA_DIR / "TMT10-Trial-8.mzML"


@pytest.fixture
def real_mzxml():
    """Get a real mzXML file."""
    return DATA_DIR / "TMT10-Trial-8.mzXML"


@pytest.fixture
def real_tdf():
    """Get a real TDF file."""
    return DATA_DIR / "test.d"


@pytest.fixture
def real_mgf():
    """Get a real MGF file."""
    return Path(__file__).parent / "../data/TMT10-Trial-8.mgf"


@pytest.fixture
def mgf_medium(tmp_path):
    """An MGF file with 100 random annotated spectra (+1 invalid)."""
    mgf_file = tmp_path / "medium.mgf"
    return _create_mgf(_random_peptides(100), mgf_file, add_problems=True)


@pytest.fixture
def mgf_small(tmp_path):
    """An MGF file with 2 annotated spectra."""
    peptides = ["LESLIEK", "EDITHR"]
    mgf_file = tmp_path / "small.mgf"
    return _create_mgf(peptides, mgf_file)


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
    missing = not charge
    charge = 2 if not charge else charge
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

    if missing:
        del mgf[3]

    return "\n".join(mgf)


def _create_mgf(peptides, mgf_file, add_problems=False, random_state=42):
    """Create a fake MGF file from one or more peptides.

    Parameters
    ----------
    peptides : str or list of str
        The peptides for which to create spectra.
    mgf_file : Path
        The MGF file to create.
    add_problems : bool
        Add weird charge states and invalid spectra.
    random_state : int or numpy.random.Generator, optional
        The random seed. The charge states are chosen to be
        2 or 3 randomly.

    Returns
    -------
    PathLike
        The MGF file.

    """
    rng = np.random.default_rng(random_state)
    peptides = list(peptides)
    entries = [_create_mgf_entry(p, rng.choice([2, 3])) for p in peptides]

    if add_problems:
        entries[-2] = _create_mgf_entry(peptides[-2], 0)
        entries[-1] = _create_mgf_entry(peptides[-1], 4)

        invalid_entry = [
            "BEGIN IONS",
            "CHARGE=2+",
            "1 1",
            "END IONS",
        ]

        entries.append("\n".join(invalid_entry))

    with mgf_file.open("w+") as mgf_ref:
        mgf_ref.write("\n".join(entries))

    return mgf_file


def _random_peptides(n_peptides, random_state=42):
    """Create random peptides.

    Parameters
    ----------
    n_peptides : int
        The number of peptides to generate.
    random_state : int or numpy.random.Generator
        The random seed or state.

    Yields
    ------
    str
        A peptide sequence

    """
    rng = np.random.default_rng(random_state)
    residues = "ACDEFGHIKLMNPQRSTUVWY"
    for i in range(n_peptides):
        yield "".join(rng.choice(list(residues), rng.integers(6, 50)))
