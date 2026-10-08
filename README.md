<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wfondrie/depthcharge/main/static/logo-dark.png">
  <img alt="depthcharge logo" src="https://raw.githubusercontent.com/wfondrie/depthcharge/main/static/logo-light.png">
</picture>

Depthcharge is a deep learning toolkit for building Transformer models to analyze mass spectrometry data.

## About

Many deep learning tools have been developed for the analysis of mass spectra or mass spectrometry analytes, like peptides and small molecules.
However, each one has had to reinvent the wheel.

Depthcharge aims to provide a flexible, but opinionated, framework for rapidly prototyping deep learning models for mass spectrometry data.
Think of Depthcharge as a set of building blocks to get you started on a new deep learning project focused around mass spectrometry data.
Depthcharge delivers these building blocks in the form of PyTorch modules, which can be readily used to assemble customized deep learning models for your task.

## Installation

Depthcharge requires Python 3.10-3.13 and can be installed from PyPI:

```sh
pip install depthcharge-ms
```

## Quick start

Parse mass spectra into a PyTorch dataset, then encode them with a Transformer:

```python
from torch.utils.data import DataLoader

import depthcharge as dc

dataset = dc.data.SpectrumDataset("spectra.mzML", batch_size=32)
model = dc.transformers.SpectrumTransformerEncoder()

for batch in DataLoader(dataset, batch_size=None):
    embeddings, mask = model(batch["mz_array"], batch["intensity_array"])
```

To learn more, visit our [documentation](https://wfondrie.github.io/depthcharge).
