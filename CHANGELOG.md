# Changelog for Depthcharge
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.4.0]

We have completely reworked of the data module.
Depthcharge now uses Apache Arrow-based formats instead of HDF5; spectra are converted either Parquet or streamed with PyArrow, optionally into Lance datasets.

We now also have full support for small molecules, with the `MoleculeTokenizer`,
`AnalyteTransformerEncoder`, and `AnalyteTransformerDecoder` classes.

### Breaking Changes
- `PeptideTransformer*` are now `AnalyteTransformer*`, providing full support for small molecule analytes. Additionally the interface has been completely reworked.
- Mass spectrometry data parsers now function as iterators, yielding batches of spectra as `pyarrow.RecordBatch` objects.
- Parsers can now be told to read arbitrary fields from their respective file formats with the `custom_fields` parameter.
- The parsing functionality of `SpctrumDataset` and its subclasses have been moved to the `spectra_to_*` functions in the data module.
- `SpectrumDataset` and its subclasses now return dictionaries of data rather than a tuple of data. This allows us to incorporate arbitrary additional data
- `SpectrumDataset` and its subclasses are now `lance.torch.data.LanceDataset` subclasses, providing native PyTorch integration.
- All dataset classes now do not have a `loader()` method.

### Added
- Support for small molecules.
- Added the `StreamingSpectrumDataset` for fast inference.
- Added `spectra_to_df`, `spectra_to_df`, `spectra_to_stream` to the `depthcharge.data` module.

### Changed
- Determining the mass spectrometry data file format is now less fragile.
  It now looks for known line contents, rather than relying on the extension.

## [v0.3.1] - 2023-08-18
### Added
- Support for fine-tuning the wavelengths used for encoding floating point numbers like m/z and intensity to the `FloatEncoder` and `PeakEncoder`.

### Fixed
- The `tgt_mask` in the `PeptideTransformerDecoder` was the incorrect type.
  Now it is `bool` as it should be.
  Thanks @justin-a-sanders!

## [v0.3.0] - 2023-06-06
### Added
- Providing a proper tokenization class (also resolves #24 and #18)
- First-class support for ProForma peptide annotations, thanks to `spectrum_utils` and `pyteomics`.
- Adding primitive dataclasses for peptides, peptide ions, mass spectra ... and even small molecules 🚀
- Adding type hints to everything and stricter linting with Ruff.
- Adding a ton of tests.
- Tight integration with `spectrum_utils` 💪

### Changed
- Moving preprocessing onto parsing instead of data loading (similar to @bittremieux's proposal in #31)
- Combining the SpectrumIndex and SpectrumDataset classes into one.
- Changing peak encodings. Instead of encoding the intensity using a linear projection and summing with the sinusoidal m/z encodings, now the intensity is also sinusoidally encoded and is combined with the sinusoidal m/z encodings using a linear layer.

## [v0.2.3] - 2023-08-18
### Fixed
- Applied hotfix from v0.3.1

## [v0.2.2] - 2023-05-15
### Fixed
- Fixed retrieving version information.

## [v0.2.1] - 2023-05-13
### Changed
- Change target mask from float to boolean.
- Log the number spectra that are skipped due to an invalid precursor charge.

## [v0.2.0] - 2023-03-06
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

## [v0.1.0] - 2022-11-15
### Changed
- The `detokenize()` method now returns a list instead of a string.

## [v0.0.1] - 2022-09-29
### Added
- This if the first release! All changes from this point forward will be
  recorded in this changelog.
