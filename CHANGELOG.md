# Changelog for Depthcharge
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added the `shuffle` and `seed` parameters to `SpectrumDataset` and `AnnotatedSpectrumDataset` (and their `from_lance()` methods), which load the mass spectra in a new random order each epoch. Blocks of `batch_size` consecutive spectra are read in a random order, then the spectra are shuffled within a buffer of 16 blocks.
- Added the `set_epoch()` method to `SpectrumDataset` and `AnnotatedSpectrumDataset`, which sets the epoch that determines the shuffled order.
- Added the `peak_file_hash` column to parsed mass spectra, which contains a fingerprint of the contents of the originating peak file. This distinguishes peak files that share the same name.
- Added `depthcharge.data.hash_peak_file()`, which quickly computes the fingerprint of a peak file from its size and first and last 1 MiB.
- Added the `peak_file_hashes` property to `SpectrumDataset` and `AnnotatedSpectrumDataset`.
- Added the `overwrite` parameter to `SpectrumDataset` and `AnnotatedSpectrumDataset`.

### Changed
- `SpectrumDataset` and `AnnotatedSpectrumDataset` now skip peak files that have already been added, as determined by their `peak_file_hash`, with a warning.
- Because of the new `peak_file_hash` column, `add_spectra()` cannot add peak files to Lance datasets that were created with previous versions of depthcharge.
- When a Lance dataset already exists at `path`, `SpectrumDataset` and `AnnotatedSpectrumDataset` now add only the peak files that it does not already contain, instead of overwriting it. This makes re-creating a dataset with the same peak files fast. Use `overwrite=True` for the previous behavior. DataFrame and parquet inputs, custom fields that are missing from the existing dataset, and datasets created with previous versions of depthcharge raise a `ValueError` unless `overwrite=True`.
- `preprocessing.scale_to_unit_norm()`, which is part of the default `preprocessing_fn`, now scales intensities to an L2 norm of 1. Previously, it divided intensities by their sum, so they summed to 1 instead. Models trained with the previous behavior may need to be retrained or use the previous function as a custom `preprocessing_fn`.
- `SpectrumDataset`, `AnnotatedSpectrumDataset`, and `StreamingSpectrumDataset` now convert batches to tensors directly from Arrow, instead of through Python objects. In our benchmarks, this makes iterating over a `SpectrumDataset` about 15 times faster. The resulting tensors and their data types are unchanged.
- `PeptideTokenizer` and `MskbPeptideTokenizer` now cache how peptide sequences are split into tokens, and all tokenizers pad tokens in a single step. This makes tokenizing previously seen peptides, such as in `AnnotatedSpectrumDataset` after the first epoch, about 9 times faster.
- `spectra_to_stream()`, and the functions and datasets that use it, now collect the `metadata_df` once, instead of once per batch. This is faster when `metadata_df` is a `polars.LazyFrame`.
- `SpectrumDataset` and `AnnotatedSpectrumDataset` now divide batches among `DataLoader` workers and distributed training processes, instead of Lance fragments. Previously, datasets with a single fragment, such as those created from a single DataFrame, were loaded entirely by one worker. Use `shard_granularity="fragment"` for the previous behavior. DataLoader workers must be started with the `"spawn"` or `"forkserver"` method, because Lance is not safe to use in forked processes.
- depthcharge now requires `pylance>=0.17.0`.
- Parsed intensities are now stored as float32 instead of float64 in the `intensity_array` column, halving their size. Intensities were already float32 precision during parsing, so no information is lost. Existing Lance datasets keep their data types, and spectra added to them are converted.
- The default `preprocessing_fn` is now implemented with NumPy. It produces the same mass spectra as before, but avoids compiling the spectrum_utils methods, which took several seconds in each new process. When several peaks have the same intensity at the cutoff for the 200 most intense peaks, a different one of them may be kept.
- mzML and mzXML files are now parsed without building an index of their spectra, which is faster.

### Fixed
- `spectra_to_stream()` no longer drops spectra when joining the `metadata_df` adds rows to a batch beyond `batch_size`, such as when a `scan_id` appears more than once.
- `add_spectra()` can now add peak files to datasets created from polars DataFrames, which previously failed because of mismatched Arrow types.

## [v0.5.0]
### Added
- Added the `replace_n_and_q_deamidated_with_d_and_e` option to the `PeptideTokenizer`, which replaces deamidated N and Q residues with D and E residues, because they are indistinguishable by de novo sequencing.
- Added the `pad_fields` option to `SpectrumDataset`, `AnnotatedSpectrumDataset`, and `StreamingSpectrumDataset` (and their `from_lance()` methods), which pads additional list columns into a single tensor for each batch.
- Added float16 and bfloat16 support to `FloatEncoder` and `PositionalEncoder`. The wavelength terms are kept at float32 precision and the encodings are cast to the model's dtype. A warning is raised for inputs with less precision than float32.

### Changed
- Spectra with invalid custom fields are now skipped and counted in the skipped spectra warning, instead of raising an error.

### Fixed
- Fixed formatting of the warning for skipped spectra (missing space and stray line break), and included the exception type in it.
- The skipped spectra warning is now raised even when iteration over a peak file stops early.
- Columns that are padded now raise an informative error when they cannot be padded, and a list column containing missing values no longer causes an error when converting a batch to tensors.
- Fixed `AnalyteTransformerDecoder.embed()` (and therefore `forward()`) failing when called with `tokens=None`, due to the empty token tensor having a float dtype and a batch size of 1.

## [v0.4.10]
### Fixed
- Changed C-terminal and N-terminal modification check to include empty modifications

## [v0.4.9]
### Added
- Added support for Bruker .d (TDF) files in parsers and datasets.
- Added the `hf_converter` keyword to `SpectrumDataset._to_tensor` and `AnnotatedSpectrumDataset._to_tensor` for pylance compatibility.
- Added Jupyter to documentation dependencies.

### Fixed
- Fixed peptide tokenizer detokenizing in reverse.
- Handle MGF `index=` prefixes during parsing.
- Handle missing precursor charge in the MzML parser.

### Changed
- Adjusted the Lance URL in datasets and documentation.

## [v0.4.8]
### Changed
- `Tokenizer.detokenize()` now truncates the output to the first stop token it finds, if `trim_stop_token=True`.

## [v0.4.7]
### Fixed
- Add stop and start tokens for `AnnotatedSpectrumDataset`, when available.
- When `reverse` is used for the `PeptideTokenizer`, automatically reverse the decoded peptide.

## [v0.4.6]
### Added
- Added support for unsigned modification masses that don't quite conform to the Proforma standard.

## [v0.4.5]
### Changed
- The `scan_id` column for parsed spectra is now a string instead of an integer. This is less space efficient, but we ran into issues with Sciex indexing when trying to use only an integer.

## [v0.4.4]

### Changed
- Partially revert length changes to `SpectrumDataset` and `AnnotatedSpectrumDataset`. We removed `__len__` from both due to problems with PyTorch Lightning compatibility.
- Simplify dataset code by removing redundancy with `lance.pytorch.LanceDatset`.
- Improved warning message for skipped spectra.

## [v0.4.3]

### Changed
- Length of the `SpectrumDataset` and `AnnotatedSpectrumDataset` now reflect the `samples` parameter of the `lance.pytorch.LanceDataset` parent class.

## [v0.4.2]

### Changed
- The length of `SpectrumDataset` and `AnnotatedSpectrumDataset` is now the number of batches, not the number of spectra. This let's tools like PyTorch Lighting create their progress bars properly.
- Parsing a dataset now no longer requires reading essentially the whole first file. Now the schema is inferred from the first 128 spectra.

## [v0.4.1]

### Added
- Significant updates to documentation. Add how to model mass spectra.
- Reading and writing from cloud storage on everything!

### Changed
- Migrated to Mike for mkdocs to manage multiple versions.
- Moved test GitHub Action from pip to uv.

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
