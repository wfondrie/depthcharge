# The depthcharge Python API

The depthcharge package provides utilities and classes to parse, store, and use mass spectra, peptides, and small molecules in PyTorch models.
Although depthcharge is primarily focused on PyTorch Transformer modules as a starting point, many of these classes and utilities can be used as building blocks for models of your own choice.

## Tokenizers

Tokenizers split string inputs into tokens, such as peptide sequences or SMILES strings, into the pieces that we want to model with our neural networks.
For example, peptide sequences need to be broken into constituent amino acids, with or without modifications (the tokens).
In addition to specifying how tokens are created, the tokenizer classes also aid in mass calculations that are useful for modeling mass spectrometry data.

Tokenizers live in the [tokenizers submodule](tokenizers).

## Data

Depthcharge provides several utilities for interacting with data such as mass spectra, small molecules and peptides.

The [data submodule](datasets) contains functions for parsing and preprocessing mass spectra.
It also contains dataset classes for loading data into models built with PyTorch.

Datasets are subclasses of PyTorch Datasets, specifically designed to store and retrieve mass spectrometry data, including the mass spectra themselves and analyte representations.

Datasets and parsing functions live in the [data submodule](datasets).

## Encoders

Often the most useful representation for an input is not the raw parsed format.
In depthcharge, we provide a collection of sinusoidal encoders to represent general floating point numbers, positions in a sequence, and peaks in a mass spectrum.

Encoders live in the [encoders submodule](encoders).

## Transformers

The Transformer architecture is a powerful tool for modeling sequence data, such as the amino acids of a peptide or even the peaks in a mass spectrum.
In depthcharge, we provide Transformer models specifically tailored for modeling different pieces of mass spectrometry and related data.
These classes are compound classes: they often combine the appropriate encoders with a traditional Transformer model to directly model the data of interest.

Transformers live in the [transformers submodule](transformers).

## Primitives

Although most users will not interact with our primitive classes directly, they are the building blocks for representing mass spectrometry data types.
These classes are not primitives in the traditional sense of the word; rather, they define how to parse and store data such as peptides, mass spectra, and small molecules.

[Primitives](primitives) are available as top-level classes in Depthcharge.
