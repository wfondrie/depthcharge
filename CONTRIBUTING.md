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

3. Install Depthcharge and its development dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev
```

4. Install the pre-commit hooks.
   These will automatically lint and verify that new code matches our standard formatting with each new commit.

```bash
uv run pre-commit install
```

5. Create a new branch with a descriptive name for your changes:

```bash
git checkout -b fix_x
```

6. Make your changes (make sure to read below first).
   Bug fixes should include a test that fails without the fix, and user-facing changes should include an entry under `## [Unreleased]` in `CHANGELOG.md`.

7. Verify that the tests pass:

```bash
uv run pytest tests/
```

8. Add, commit, and push your changes to your forked repository.

9. On the GitHub page for you forked repository, click "Pull request" to propose
   adding your changes to Depthcharge.

10. We'll review, discuss, and help you make any revisions that are required. If
   all goes well, your changes will be added to Depthcharge
   in the next release!

If you are using an AI coding agent, point it to [AGENTS.md](https://github.com/wfondrie/depthcharge/blob/main/AGENTS.md), which describes our conventions in more detail.


### Python code style

The Depthcharge project follows the [PEP 8 guidelines](https://www.python.org/dev/peps/pep-0008/) for Python code style.
More specifically, we use [Ruff](https://docs.astral.sh/ruff/) to automatically format and lint Python code in Depthcharge.
All functions, classes, and methods should have NumPy-style docstrings.

We highly recommend setting up our pre-commit hooks (step 4 above).
These will run Ruff and some other checks during each commit, fixing problems that can be fixed automatically.
Because we check formatting and linting as part of our tests, setting up these hooks can save you from having to revise your code.
If a hook changes a file, you need to `git add` the file again before finishing the commit.

You can also run Ruff yourself:

```bash
uv run ruff check --fix .
uv run ruff format .
```

When you're ready, open a pull request with your changes and we'll start the review process.
Thank you for your contribution! :tada:
