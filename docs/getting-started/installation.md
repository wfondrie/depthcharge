# Installing Depthcharge

## Requirements

### Python
Depthcharge is a Python package and requires Python >= 3.10.
To check what version of Python you have installed, you can run:

``` sh
$ python --version
```

If you do not have Python installed, we recommend Miniconda, which also comes with the conda package manager.
See the [Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html) for details.

### PyTorch (*optional*)
Depthcharge creates PyTorch modules which you can use to create deep learning models for your application.
PyTorch can be installed automatically during Depthcharge installation; however, to get the most out of Depthcharge and PyTorch, you'll likely want to install PyTorch manually to account for any GPUs and their drivers that you may have installed.
To install PyTorch, follow the installation instructions found in the [PyTorch documentation](https://pytorch.org/get-started/locally/).


## Install depthcharge

Depthcharge can be install with `pip`, directly from PyPI:
```sh
$ pip install depthcharge-ms
```

Eventually Depthcharge will be made available through Bioconda.
