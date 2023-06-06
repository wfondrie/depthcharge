# Contributing to depthcharge

First off, thank you for taking the time to contribute.

The following document provides guidelines for contributing to the
documentation and the code of Depthcharge. **No contribution is too small!** Even
fixing a simple typo in the documentation is immensely helpful.


## Contributing to the documentation

We use [mkdocs](https://www.mkdocs.org/) generate our
documentation and deploy it to this site. Most of the pages on the site are
created from simple text files written in the Markdown markup language.
There are three exceptions to this:

1. The API.

2. The Vignettes are created from Jupyter notebooks.

3. The Code of Conduct, Release Notes, Changlog, and this Contributing document are
   markdown files that live in the root of the Depthcharge repository.

### Editing most documents

The easiest way to edit a document is by clicking the "Edit on GitHub" like in
the top right hand corner of each page. You'll be taken to GitHub where
you can click on the pencil to edit the document.

You can then make your changes directly on GitHub. Once you're finished, fill
in a description of what you changed and click the "Propose Changes" button.

Alternatively, these documents live in the `docs/` directory of the
repository and can be edited like code. See [Contributing to the
code](#contributing-to-the-code) below for more details on contributing this
way.


## Contributing to the code

We welcome contributions to the source code of Depthcharge---particularly
ones that address discussed [issues](https://github.com/wfondrie/depthcharge/issues).

Contributions to Depthcharge follow a standard GitHub contribution workflow:

1. Create your own fork of the Depthcharge repository on GitHub.

2. Clone your forked Depthcharge repository to work on locally.

2. Install the pre-commit hooks.
   These will automatically lint and verify that new code matches our standard formatting with each new commit.
```bash
# If you need to install pre-commit:
pip install pre-commit

# Install the pre-commit hooks:
pre-commit install
```

3. Create a new branch with a descriptive name for your changes:

```bash
git checkout -b fix_x
```

4. Make your changes (make sure to read below first).

5. Add, commit, and push your changes to your forked repository.

6. On the GitHub page for you forked repository, click "Pull request" to propose
   adding your changes to Depthcharge.

7. We'll review, discuss, and help you make any revisions that are required. If
   all goes well, your changes will be added to Depthcharge
   in the next release!


### Python code style

The Depthcharge project follows the [PEP 8 guidelines](https://www.python.org/dev/peps/pep-0008/) for Python code style.
More specifically, we use [Black](https://black.readthedocs.io/en/stable/) to automatically format code and [Ruff](https://github.com/charliermarsh/ruff) to automatically lint Python code in Depthcharge.

We highly recommend setting up our pre-commit hooks.
These will run Black, Ruff, and some other checks during each commit, fixing problems that can be fixed automatically.
Because we run black for code linting as part of our tests, setting up this hook can save you from having to revise code formatting. Take the following steps to setup the pre-commit hooks:

1. Verify that pre-commit is installed on your machine.
   If not, you can install them with pip or conda:

```bash
# Using pip
pip install pre-commit

# Using conda
conda -c conda-forge pre-commit
```

2. Navigate to your local copy of the Depthcharge repository and activate the hook:
```bash
pre-commit install
```

One the hook is installed, black will be run before any commit is made. If a
file is changed by black, then you need to `git add` the file again before
finished the commit.

When you're ready, open a pull request with your changes and we'll start the review process.
Thank you for your contribution! :tada:
