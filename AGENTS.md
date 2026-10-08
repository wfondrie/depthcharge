# AGENTS.md

Guidance for AI coding agents working on depthcharge.
Human contributors should also read [CONTRIBUTING.md](CONTRIBUTING.md).

## Project overview

depthcharge (published on PyPI as `depthcharge-ms`) is a toolkit for building
deep learning models of mass spectrometry data with PyTorch, with a focus on
Transformers.

- `depthcharge/data/`: Peak file parsers (mzML, mzXML, MGF, Bruker TDF),
  Arrow/parquet conversion, and PyTorch datasets built on Lance.
- `depthcharge/tokenizers/`: Peptide and small molecule tokenizers.
- `depthcharge/encoders/`: Sinusoidal encoders for m/z values and positions.
- `depthcharge/transformers/`: Transformer encoders and decoders for spectra
  and analytes.
- `depthcharge/primitives.py`: Core objects, such as `MassSpectrum` and
  `Peptide`.
- `tests/unit_tests/`: Tests, mirroring the package layout. Shared fixtures
  live in `tests/conftest.py`, and test peak files live in `data/`.
- `docs/`: The mkdocs site. API pages in `docs/api/` use mkdocstrings
  (`::: depthcharge.module.Object`).

## Setup and commands

We use [uv](https://docs.astral.sh/uv/). Python 3.10-3.13 is supported.

```bash
uv sync --dev                # Install the package and dev dependencies
uv run pytest tests/         # Run the tests
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run pre-commit install    # Install the pre-commit hooks
```

To build the docs, install the docs extra (`uv sync --extra docs`) and
[Quarto](https://quarto.org), then run `uv run mkdocs serve`.

Before you consider a change finished, `ruff check .`, `ruff format --check .`,
and `pytest tests/` must all pass. CI runs the tests on Linux, macOS, and
Windows.

## Code conventions

- Formatting and linting are done with Ruff (not Black), using a line length
  of 79. The rules are configured in `pyproject.toml`.
- Every module, class, and function, including private helpers, has a
  NumPy-style docstring with `Parameters` and `Returns` (or `Yields`)
  sections, as in the existing code.
- Use type hints on all function signatures.
- Do not use `print()`. Use `logging` or `warnings`.
- For arguments that accept one item or many, follow the existing pattern
  of accepting `X | Iterable[X]` and normalizing with `utils.listify()`.
- Use `pathlib` for paths. Do not assume a POSIX file system.
- When you add public API:
  - Export it from the subpackage's `__init__.py`.
  - Add it to the matching page in `docs/api/`.
  - Update the docstrings of any parameters that you change.
- Keep changes scoped to the task. Do not reformat or refactor unrelated
  code.

## Tests

- Every bug fix needs a regression test that fails without the fix.
- Every new feature needs tests that cover each code path that uses it. For
  example, datasets have `SpectrumDataset`, `from_lance()`, and
  `StreamingSpectrumDataset` paths.
- For code that handles tensors, test with a batch size greater than 1.
  Shape bugs often only appear with multiple examples.
- Use pytest's `tmp_path` for files, and the fixtures in `tests/conftest.py`
  for peak files.
- Use `pytest.warns` and `pytest.raises` with `match=` to check the content
  of warnings and errors.

## Changelog

Every user-facing change needs an entry in `CHANGELOG.md` under
`## [Unreleased]`. Entries go under `### Added`, `### Changed`, or
`### Fixed`, following [Keep a Changelog](https://keepachangelog.com).

- Name the public API that changed, such as
  ``Added the `pad_fields` option to `SpectrumDataset` ``.
- Put changes in existing behavior under `### Changed`, even if they are
  improvements, so that users can find them when upgrading.

## Versions and releases

- The version comes from git tags through `setuptools_scm`. Do not hard-code
  or edit version numbers.
- Releases are made by maintainers only. Creating a GitHub release publishes
  the package to PyPI and deploys the docs. Agents must not create tags or
  releases.

## Git and pull requests

- Work on a branch, and open pull requests against `main`. Pull requests
  are squash-merged.
- Contributor branches may live on forks. To update one with `main`, merge
  `main` into it rather than rebasing, so that no force push is needed.
- When you resolve merge conflicts, never restore files from `HEAD` (the
  pre-merge commit) while a merge is in progress. Doing so silently reverts
  every upstream change to those files. After any merge, confirm that
  `git diff origin/main --stat` lists only the changes you intended.
- Write pull request descriptions that explain what changed and why. Call
  out any behavior changes.
