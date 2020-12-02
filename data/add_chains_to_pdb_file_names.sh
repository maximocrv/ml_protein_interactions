#!/bin/sh

for f in pdbs_wt/*.pdb; do
	b=$(echo $f | cut -d'/' -f2 | cut -d'.' -f1)
	new_fname=$(awk "BEGIN{FS=\";\"}/^$b/{print \$1; exit}" skempi_v2.csv)
	[ ! -f "pdbs_wt/$new_fname.pdb" ] && { 
		echo "$b -> $new_fname"
		mv $f "pdbs_wt/$new_fname.pdb"
		# mv "pdbs_wt/$b.mapping" "pdbs_wt/$new_fname.mapping"
	}
done
