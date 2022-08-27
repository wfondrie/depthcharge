![depthcharge logo](./static/logo-dark.png#gh-dark-mode-only)
![depthcharge logo](./static/logo-light.png#gh-light-mode-only)

Depthcharge is a deep learning toolkit for building state-of-the-art models to analyze mass spectra generated from peptides other and molecules.

## About

Many deep learning tools have been developed for the analysis of mass spectra, particularly for proteomics and metabolomics data (Prosit, MS2Pip, DeepNovo, pNovo, etc.). However, each one has had to reinvent the wheel. 
Depthcharge aims to be a general framework for creating state-of-the-art deep learning models for mass spectrometry data, empowering developers and researchers to spend more time on innovating rather than recreating the layers needed for handling mass spectra and peptides/other molecules. 
Think of depthcharge as a set of building blocks to get you stared on a new deep learning project focused around mass spectrometry data. 
Depthcharge delivers these building blocks mostly in the form of PyTorch modules, which can be readily used to assemble customized deep learning models for your task.

Currently, depthcharge focuses on Transformer layers, a type of model that has been revolutionary for natural language processing (NLP) and computer vision tasks.
We've found Transformers to be particularly well-suited for mass spectra and our *de novo* peptide sequencing tool, [Casanovo](https://github.com/Noble-Lab/casanovo), is built upon depthcharge.


## Installation

Depthcharge can currently be installed with pip, and will eventually be made available through [Bioconda](https://bioconda.github.io).

### Requirements

#### Python
Depthcharge is a Python package and requires Python >= 3.8. 
To check what version of Python you have installed, you can run:

``` sh
$ python --version
```

If you do not have Python installed, we recommend Miniconda, which also comes with the conda package manager. 
See the [Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html) for details. 

#### PyTorch (*optional*)
Depthcharge creates PyTorch modules which you can use to create deep learning models for your application. 
PyTorch can be installed automatically during Depthcharge installation; however, to get the most out of Depthcharge and PyTorch, you'll likely want to install PyTorch manually to account for any GPUs and their drivers that you may have installed. 
To install PyTorch, follow the installation instructions found in the [PyTorch documentation](https://pytorch.org/get-started/locally/).

#### ppx (*optional*)
ppx is a Python package for downloading data from the MassIVE and PRIDE repositories.
Although Depthcharge does not depend on ppx, we do use it in our vignettes.
To install ppx, either use pip:

``` sh
$ pip install ppx
```

or conda:

``` sh
$ conda install -c bioconda ppx
```


### Install with pip
Install depthcharge with pip:

``` sh
$ pip install depthcharge-ms
```

### Install with conda
Conda installations is not yet available, but will eventually be possible using the [Bioconda](https://bioconda.github.io) channel.


---
More documentation coming soon!

<style>
  .md-typeset h1 {
    visibility: hidden;
    font-size: 2px; 
    margin: 0px;
  }
</style>

