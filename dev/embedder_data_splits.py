#!/usr/bin/etc python3
"""Manage the projects in each data split for the embedding model.

The entire point of this module is to persist the training, validation, and
test sets.
"""
import re
import json
import logging
import subprocess
from pathlib import Path
from argparse import ArgumentParser
from tempfile import TemporaryDirectory

import ppx
import numpy as np
import pandas as pd

SPLITS = Path(__file__).parent / "../depthcharge/embedder/SPLITS.json"
LOGGER = logging.getLogger(__name__)


class ProjectSplits:
    """A class to manage data splits for the embedding task.

    Parameters
    ----------
    n_train : int, optional
        The number of projects to use for training.
    n_valid : float, optional
        The number of projects to use for validation.
    n_test : float, optional
        The number of projects to use for testing.
    path : Path, optional
        The path to the JSON file storing previously used files.
    projects : list of str
        The projects to choose from. By default this is all of MassIVE.
    random_state : int or Generator, optional
        The numpy random state.
    """

    def __init__(
        self,
        n_train=50,
        n_valid=7,
        n_test=7,
        path=None,
        project=None,
        random_state=42,
    ):
        """Initialize the ProjectSplits"""
        self.rng = np.random.default_rng(random_state)
        if path is None:
            self._path = SPLITS
        else:
            self._path = Path(path)

        if self.path.exists():
            with self.path.open("r") as split_data:
                self.splits = json.load(split_data)
        else:
            self.splits = {
                "train": {},
                "validation": {},
                "test": {},
                "rejected": [],
            }

        # Get the projects to choose from:
        used = []
        for projects in self.splits.values():
            if not projects:
                continue

            try:
                used += list(projects.keys())
            except AttributeError:
                used += projects

        if projects is None:
            projects = ppx.massive.list_projects()

        avail = np.array([p for p in projects if p not in used])
        self.rng.shuffle(avail)
        self._projects = list(avail)

        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test

    def save(self):
        """Update the underlying JSON file."""
        with self.path.open("w+") as split_data:
            json.dump(self.splits, split_data, indent=4)

    def add_projects(self, split, num):
        """Add random projects to a split.

        Parameters
        ----------
        split : str, {"train", "validation", "test"}
            The split to add projects to.
        num : int
            The number of random projects to add.
        """
        added = 0
        pattern = re.compile(r"ccms_peak/.*\.mzml$", flags=re.IGNORECASE)
        for idx, msvid in enumerate(self._projects):
            proj = ppx.find_project(msvid)
            keep = []
            file_info = [l.split(",") for l in proj.file_info().splitlines()]
            ms2_idx = file_info[0].index("spectra_ms2")
            for info in file_info[1:]:
                fname = info[0].split("/", 1)[1]
                if pattern.search(fname) and int(info[ms2_idx]):
                    keep.append(fname)

            if keep and validate(proj, keep[0]):
                self.splits[split][msvid] = keep
                added += 1
                LOGGER.info("Found %i/%i...", added, num)
            else:
                self.splits["rejected"].append(msvid)

            if added == num:
                break

            self.save()

        # Remove the projects we've sampled from consideration:
        del self._projects[: idx + 1]
        if added < num:
            LOGGER.warn("Not enough projects for the request. Added %i", added)

        return added

    def purge_test_set(self):
        """Add the current test set to the training set."""
        self.splits["validation"].update(self.splits["test"])
        self.splits["test"] = {}
        self.n_test = 0

    @property
    def n_train(self):
        """The number of training projects"""
        return self._n_train

    @n_train.setter
    def n_train(self, num):
        """Set the number of training projects"""
        curr = len(self.splits["train"])
        if curr > num:
            LOGGER.warn(
                "More training projects have already been added. "
                "Setting to %i.",
                curr,
            )
            self._n_train = curr
        elif curr == num:
            self._n_train = curr
        else:
            LOGGER.info("Adding training projects...")
            added = self.add_projects("train", num - curr)
            self._n_train = added

    @property
    def n_valid(self):
        """The number of validation projects"""
        return self._n_valid

    @n_valid.setter
    def n_valid(self, num):
        """Set the number of validation projects"""
        curr = len(self.splits["validation"])
        if curr > num:
            LOGGER.warn(
                "More validation projects have already been added. "
                "Setting to %i.",
                curr,
            )
            self._n_valid = curr
        elif curr == num:
            self._n_valid = curr
        else:
            LOGGER.info("Adding validation projects...")
            added = self.add_projects("validation", num - curr)
            self._n_valid = added

    @property
    def n_test(self):
        """The number of test projects"""
        return self._n_test

    @n_test.setter
    def n_test(self, num):
        """Set the number of validation projects"""
        curr = len(self.splits["test"])
        if curr > num:
            LOGGER.warn(
                "More testing projects have already been added. "
                "Setting to %i.",
                curr,
            )
            self._n_test = curr
        elif curr == num:
            self._n_test = curr
        else:
            LOGGER.info("Adding testing projects...")
            added = self.add_projects("test", num - curr)
            self._n_test = added

    @property
    def path(self):
        """The path to the JSON file."""
        return self._path


def validate(proj, fname, frag_tol=0.05):
    """Verify the project has high-resolution data using param-medic

    Parameters
    ----------
    proj : str
        The ppx.Project.
    fname : str
        The remote mzML file name.
    frag_tol: float
        The minimum acceptible fragment tolerance from param-medic

    Returns
    -------
    bool
        Proceed with the project?
    """
    downloaded = proj.download(fname)[0]
    pm_tol = param_medic(downloaded)
    if pm_tol is None or pm_tol > frag_tol:
        downloaded.unlink()
        return False

    return True


def param_medic(mzml_file):
    """Run param-medic and return the fragment bin size.

    Parameters
    ----------
    mzml_file : str
        The mzML file to run param-medic on.

    Returns
    -------
    float or None
        The fragment bin size or None if param-medic failed.
    """
    with TemporaryDirectory() as tmp:
        cmd = [
            "crux",
            "param-medic",
            "--output-dir",
            tmp,
            str(mzml_file),
        ]

        out_file = Path(tmp, "param-medic.txt")

        try:
            subprocess.run(cmd, check=True)
            tol = pd.read_table(out_file).loc[0, "fragment_prediction_th"]
        except (pd.errors.EmptyDataError, subprocess.CalledProcessError):
            # param-medic failed on a previous run.
            tol = None

    if tol == "ERROR":
        tol = None

    return tol


def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(
        description="""
        Manage data splits for the GSP embedding model. This script controls
        the MassIVE projects that are included in the training, validation, and
        test splits. One file from each projects is tested to see if it is
        high-resolution using param-medic, which is a tool in the crux mass
        spectrometry tool kit. Thus, crux will need to be in your path to run
        this script. You can download it here: http://crux.ms/

        By default projects are added to a JSON file within depthcharge, with
        the intention that projects are only ever added (not removed).
        """
    )

    parser.add_argument(
        "--n_train",
        default=40,
        type=int,
        help="The number of training projects.",
    )

    parser.add_argument(
        "--n_valid",
        default=5,
        type=int,
        help="The number of validation projects.",
    )

    parser.add_argument(
        "--n_test",
        default=5,
        type=int,
        help="The number of test projects",
    )

    parser.add_argument(
        "--purge_test_set",
        action="store_true",
        help="""
        Get new projects for the test set. Previous files will be added to the
        training set.
        """,
    )

    parser.add_argument(
        "--path",
        type=str,
        help="The JSON file in which to record the data splits.",
    )

    return parser.parse_args()


def main():
    """The main function"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s]: %(message)s",
    )

    args = parse_args()
    splits = ProjectSplits(
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_test=args.n_test,
        path=args.path,
    )

    if args.purge_test_set:
        splits.purge_test_set()
        splits.n_test = args.n_test

    splits.save()
    LOGGER.info("DONE!")


if __name__ == "__main__":
    main()
