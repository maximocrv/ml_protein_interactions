# ml_protein_protein

Machine learning project investigating protein-protein interactions. 

Install [Anaconda/miniconda (latter takes less time to install)](https://docs.conda.io/en/latest/miniconda.html) as python distribution.

Install the env with the following command:
```
conda env create -f environment.yml
```

now activate the conda environment by typing 
```
conda activate ml_protein_protein
```

If you want to make sure your files are pep8 compliant before commiting then create a pre-commit file in .git/hooks/ 
with the following content
```
#!/bin/sh
set -e

flake8 --max-line-length=120
```