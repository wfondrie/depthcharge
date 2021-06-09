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
        "G": 57.02146,
        "A": 71.03711,
        "S": 87.03203,
        "P": 97.05276,
        "V": 99.06841,
        "T": 101.04768,
        "C": 103.00919,
        "L": 113.08406,
        "I": 113.08406,
        "N": 114.04293,
        "D": 115.02694,
        "Q": 128.05858,
        "K": 128.09496,
        "E": 129.04259,
        "M": 131.04049,
        "H": 137.05891,
        "F": 147.06841,
        "U": 150.95364,
        "R": 156.10111,
        "Y": 163.06333,
        "W": 186.07931,
        "O": 237.14773,
    }

    # Modfications found in MassIVE-KB
    _extended = {
        # N-terminal mods:
        "+42.011": 42.010565,  # Acetylation
        "+43.006": 43.005814,  # Carbamylation
        "-17.027": -17.026549,  # NH3 loss
        "+43.006-17.027": (43.006814 - 17.026549),
        # AA mods:
        "M+15.995": 15.994915,  # Met Oxidation
        "N+0.984": 0.984016,  # Asn Deamidation
        "Q+0.984": 0.984016,  # Gln Deamidation
    }

    proton = 1.007276
    h2o = 2 * proton + 15.994915

    def __init__(self, extended=False):
        """Initialize the PeptideMass object"""
        self.masses = self._aa_mass
        if extended:
            self.masses = self.masses.update(self._extended)

    def __len__(self):
        """Return the length of the residue dictionary"""
        return len(self.masses)

    def mass(self, seq, charge=None):
        """Calculate a peptide's mass"""
        calc_mass = sum([self.masses[aa] for aa in seq]) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass
