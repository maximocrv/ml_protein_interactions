
remove-env:
	conda env remove -n ml_protein_protein
install-env:
	conda env create -f environment.yml
reinstall-env: remove-env install-env
update-env:
	conda env update --file environment.yml  --prune
