"""Amino acid masses and other useful mass spectrometry calculations"""
import re


class PeptideMass:
    """A simple class for calculating peptide masses

    Parameters
    ----------
    extended: bool, optional
        Include the modifications present in MassIVE-KB.
    """

    _aa_mass = {
        "G": 57.021463735,
        "A": 71.037113805,
        "S": 87.032028435,
        "P": 97.052763875,
        "V": 99.068413945,
        "T": 101.047678505,
        "C": 103.009184505,
        "L": 113.084064015,
        "I": 113.084064015,
        "N": 114.042927470,
        "D": 115.026943065,
        "Q": 128.058577540,
        "K": 128.094963050,
        "E": 129.042593135,
        "M": 131.040484645,
        "H": 137.058911875,
        "F": 147.068413945,
        "U": 150.953633405,
        "R": 156.101111050,
        "Y": 163.063328575,
        "W": 186.079312980,
        "O": 237.147726925,
    }

    # Modfications found in MassIVE-KB
    _extended = {
        # N-terminal mods:
        "+42.011": 42.010565,  # Acetylation
        "+43.006": 43.005814,  # Carbamylation
        "-17.027": -17.026549,  # NH3 loss
        "+43.006-17.027": (43.006814 - 17.026549),
        # AA mods:
        "M+15.995": _aa_mass["M"] + 15.994915,  # Met Oxidation
        "N+0.984": _aa_mass["N"] + 0.984016,  # Asn Deamidation
        "Q+0.984": _aa_mass["Q"] + 0.984016,  # Gln Deamidation
        "C+57.021": _aa_mass["C"] + 57.02146,
    }

    hydrogen = 1.007825035
    oxygen = 15.99491463
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688

    def __init__(self, extended=False):
        """Initialize the PeptideMass object"""
        self.masses = self._aa_mass
        if extended:
            self.masses.update(self._extended)

    def __len__(self):
        """Return the length of the residue dictionary"""
        return len(self.masses)

    def mass(self, seq, charge=None):
        """Calculate a peptide's mass"""
        calc_mass = sum([self.masses[aa] for aa in seq]) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass
