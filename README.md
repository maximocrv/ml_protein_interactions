## ml_protein_protein

The objective of this project is to reliably predict the relative binding affinity arising due to mutations in protein structures, which is an active field of research in protein engineering. 

## File Structure

* `data`: This directory contains all the data required for training and validation.

* `notebooks`: It contains the data_exploration.ipynb jupyter notebook which is used for visualising and exploring the dataset for this project. 

* `scripts`: This directory contains all the Python scripts used for data acquisition, preprocessing, training and validation.


## Requirements

* Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (latter takes less time to install) as your python distribution.

> **Note**: due to the `pdbfixer` and `openmm` packages, this project is running on Python 3.7.

> **Note**: having an Nvidia GPU available is highly recommended, although not necessary.

## Installation

Install the env with the following command:
```
conda env create -f environment.yml
```

Now activate the conda environment by typing:
```
conda activate ml_protein_protein
```

If you need to update the environment use:
```
conda env update -f environment.yml
```

If for whatever reason you need to delete the environment due to conflicts, you can run:
```
conda env remove -n ml_protein_protein
```

Thereafter you have to run `conda deactivate` and you will be able to reinstall everything properly. 

> A `Makefile` already provides these commands for faster access. Simply run `make install-env`.

If you want to make sure your files are pep8 compliant before committing then create a pre-commit file in .git/hooks/
with the following content:
```
#!/bin/sh
set -e

flake8 --max-line-length=120
```
