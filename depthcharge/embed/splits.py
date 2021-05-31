"""Manage the projects in each data split.

The entire point of this module is to persist the training, validation, and
test sets.
"""
import random
from pathlib import Path

import pandas as pd


def get_splits():
    """Get the files in the training, validation, and test sets.

    These splits are created by project.

    Returns
    -------
    training_set : list of str
        The projects in the training set
    validation_set : list of str
        The projects in the validation set
    test_set : list of str
        The projects in the validation set
    """
    project_df = pd.read_csv("data/splits.csv")
    projects = []
    for split in ["train", "validation", "test"]:
        split_proj = project_df.loc[project_df["split"] == split, "project"]
        projects.append(split_proj.to_list())

    return tuple(projects)


def add_projects(projects, train=0.7, validation=0.1, test=0.2):
    """Add projects maintaining the provided ratios

    Projects will be added to previous splits in the ratios provided.

    Parameters
    ----------
    projects : list of str
        The projects to add.
    train : float
        The proportion of projects to add to the training set.
    validation : float
        The proportion of projects to add to the validation set.
    test : float
        The proportion of projects to add to the test set.
    """
    coeff = 1 / (train + validation + test)
    train = int(len(projects) // (1 / (coeff * train)))
    validation = int(len(projects) // (1 / (coeff * validation)))
    test = len(projects) - train - validation

    random.shuffle(projects)
    df = pd.DataFrame({"projects": projects})
    df["split"] = (
        ["train"] * train + ["validation"] * validation + ["test"] * test
    )

    if Path("data/splits.csv").exists():
        df = pd.concat([pd.read_csv("data/splits.csv"), df])

    df.to_csv("data/splits.csv", index=False)


def purge_test_set():
    """Move current test set projects to the training set.

    This is useful to refresh the test set after its been used.
    """
    df = pd.read_csv("data/splits.csv")
    df.loc[df["split"] == "test", "split"] = "train"
    df.to_csv("data/splits.csv", index=False)
