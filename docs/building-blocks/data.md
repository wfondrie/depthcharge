# Data Handling

## Mass Spectrometry Data

Depthcharge includes classes to parse and store mass spectra from mzML or MGF
files in an accessible HDF5 format, which we refer to as an index.

::: depthcharge.data.SpectrumIndex
::: depthcharge.data.AnnotatedSpectrumIndex

## PyTorch Datasets

These classes are `torch.utils.data.Dataset` that will function with
PyTorch DataLoaders.

::: depthcharge.data.SpectrumDataset
::: depthcharge.data.AnnotatedSpectrumDataset

## PyTorch-Lightning LightningDataModules

PyTorch-Lightning data modules allow you to easily manage training, validation, and test data with their associated data loaders in an organized way.

::: depthcharge.data.SpectrumDataModule


## Preprocessing Functions

::: depthcharge.data.preprocessing
