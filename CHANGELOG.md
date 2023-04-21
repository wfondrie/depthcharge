# Changelog for depthcharge
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Log the number spectra that are skipped due to an invalid precursor charge.

## [0.2.0] - 2023-03-06
### Breaking Changes
- Dropped pytorch-lightning as a dependency.
- Removed SpectrumDataModule
- Removed full-blown models (depthcharge.models)
- Fixed sinusoidal encoders (Issue #27)
- `MassEncoder` is now `FloatEncoder`, because its generally useful for encoding floating-point numbers.

### Added
- pre-commit hooks and linting with Ruff.

### Changed
- Tensorboard is now an optional dependency.

### Removed
- The example de novo peptide sequencing model.

## [0.1.0] - 2022-11-15
### Changed
- The `detokenize()` method now returns a list instead of a string.

## [0.0.1] - 2022-09-29
### Added
- This if the first release! All changes from this point forward will be
  recorded in this changelog.
