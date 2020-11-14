
install-env:
	conda env create -f environment.yml
update-env:
	conda env update --file environment.yml  --prune
