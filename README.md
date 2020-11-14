# ml_protein_protein

Machine learning project investigating protein-protein interactions.

Install [Anaconda/miniconda (latter takes less time to install)](https://docs.conda.io/en/latest/miniconda.html) as python distribution.

Note that due to the `pdbfixer` and `openmm` packages, this project is running on Python 3.7.

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

If you want to make sure your files are pep8 compliant before commiting then create a pre-commit file in .git/hooks/
with the following content
```
#!/bin/sh
set -e

flake8 --max-line-length=120
```
