"""Test that peptide masses are calculated correctly."""
import math
from functools import partial

from pyteomics import mass
from depthcharge.masses import PeptideMass


def test_string():
    """Test peptide strings"""
    pep = PeptideMass(residues="massivekb")
    aa_mass = dict(mass.std_aa_mass)
    aa_mass["a"] = 42.010565
    aa_mass["o"] = 15.994915

    pymass = partial(
        mass.fast_mass,
        ion_type="M",
        aa_mass=aa_mass,
    )

    isclose = partial(math.isclose, rel_tol=1e-6)

    seq = "LESLIEK"
    assert isclose(mass.calculate_mass(seq), mass.fast_mass(seq))

    assert isclose(pep.mass(seq), pymass(seq))
    assert isclose(pep.mass(seq, 1), pymass(seq, charge=1))
    assert isclose(pep.mass(seq, 2), pymass(seq, charge=2))
    assert isclose(pep.mass(seq, 3), pymass(seq, charge=3))

    seq = "+42.011LESLIEM+15.995K"
    seq2 = "aLESLIEMoK"
    assert isclose(pep.mass(seq), pymass(seq2))
    assert isclose(pep.mass(seq, 1), pymass(seq2, charge=1))
    assert isclose(pep.mass(seq, 2), pymass(seq2, charge=2))
    assert isclose(pep.mass(seq, 3), pymass(seq2, charge=3))


def test_list():
    """Test peptides as a list."""
    pep = PeptideMass(residues="massivekb")
    aa_mass = dict(mass.std_aa_mass)
    aa_mass["a"] = 42.010565
    aa_mass["o"] = 15.994915

    pymass = partial(
        mass.fast_mass,
        ion_type="M",
        aa_mass=aa_mass,
    )

    isclose = partial(math.isclose, rel_tol=1e-6)

    seq = "LESLIEK"
    lseq = list(seq)
    assert isclose(mass.calculate_mass(lseq), mass.fast_mass(seq))

    assert isclose(pep.mass(lseq), pymass(seq))
    assert isclose(pep.mass(lseq, 1), pymass(seq, charge=1))
    assert isclose(pep.mass(lseq, 2), pymass(seq, charge=2))
    assert isclose(pep.mass(lseq, 3), pymass(seq, charge=3))

    seq = "aLESLIEMoK"
    lseq = ["+42.011", "L", "E", "S", "L", "I", "E", "M+15.995", "K"]
    assert isclose(pep.mass(lseq), pymass(seq))
    assert isclose(pep.mass(lseq, 1), pymass(seq, charge=1))
    assert isclose(pep.mass(lseq, 2), pymass(seq, charge=2))
    assert isclose(pep.mass(lseq, 3), pymass(seq, charge=3))
