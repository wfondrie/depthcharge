"""Manage the projects in each data split.

The entire point of this module is to persist the training, validation, and
test sets.
"""
import random
from pathlib import Path

import pandas as pd

SPLITS = Path(__file__).parent / "SPLITS.csv"


def get_splits():
    """Get the files in the training, validation, and test sets.

    These splits are created by project. Note that we expect the cached
    spectrum files to be in the same directory and to use a file root that
    indicates the project from which they originated.

    Parameters
    ----------
    npy_dir : str or Path
        The directory containing the cached spectrum files.

    Returns
    -------
    training_set : list of str
        The files in the training set
    validation_set : list of str
        The files in the validation set
    test_set : list of str
        The files in the validation set
    """
    proj_df = pd.read_csv(SPLITS)
    out = []
    for split in ["train", "validation", "test"]:
        files = proj_df.loc[proj_df["split"] == split, "index_file"].to_list()
        out.append(files)

    return tuple(out)


def add_files(
    index_files,
    train=0.7,
    validation=0.1,
    test=0.2,
):
    """Add files, maintaining splits by project.

    New projects will be added to previous splits in the ratios provided.
    The index files should contain a prefix indicating the project from
    which they are derived.

    Parameters
    ----------
    index_files : list of Path
        The index files to add.
    train : float, optional
        The proportion of projects to add to the training set.
    validation : float, optional
        The proportion of projects to add to the validation set.
    test : float, optional
        The proportion of projects to add to the test set.

    """
    index_files = [Path(f).name for f in index_files]
    if SPLITS.exists():
        proj_dict = (
            pd.read_csv(SPLITS)
            .loc[:, ["split", "project"]]
            .drop_duplicates()
            .set_index("project")
            .to_dict()["split"]
        )
    else:
        proj_dict = {}

    new = pd.DataFrame({"index_file": index_files})
    new["project"] = new["index_file"].str.split(".", expand=True).iloc[:, 0]
    new["split"] = new["project"].map(proj_dict)

    new_proj = (
        new["project"]
        .loc[pd.isna(new["split"])]
        .drop_duplicates()
        .sample(frac=1)
        .to_list()
    )

    # Assign new projects to splits:
    coeff = 1 / (train + validation + test)
    if train:
        train = int(len(new_proj) // (1 / (coeff * train)))

    if validation:
        validation = int(len(new_proj) // (1 / (coeff * validation)))

    if test:
        test = len(new_proj) - train - validation

    split = ["train"] * train + ["validation"] * validation + ["test"] * test
    new_proj_dict = {k: v for k, v in zip(new_proj, split)}

    idx = pd.isna(new["split"])
    new.loc[idx, "split"] = new.loc[idx, "project"].map(new_proj_dict)

    if SPLITS.exists():
        new = pd.concat([pd.read_csv(SPLITS), new])

    new = new.drop_duplicates()[["split", "project", "index_file"]]
    new.to_csv(SPLITS, index=False)


def purge_test_set():
    """Move current test set projects to the training set.

    This is useful to refresh the test set after its been used.
    """
    df = pd.read_csv(SPLITS)
    df.loc[df["split"] == "test", "split"] = "train"
    df.to_csv(SPLITS, index=False)
