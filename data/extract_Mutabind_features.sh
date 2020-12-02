#!/bin/sh


FOUND_N=0
NOT_FOUND_N=0
while read -r wt mut; do
	mut=$(echo "$mut" | tr ',' '_' | cut -d'"' -f2)
	fname="${wt}_${mut}"
	if $(ls pdbs_mutated | grep -q "$fname"); then
		FOUND_N=$((FOUND_N+1))
		echo "found $fname"
	else
		NOT_FOUND_N=$((NOT_FOUND_N+1))
		echo "not found $fname"
	fi
done <<EOF
$(awk '//{if(NR>1) print $1"_"$2"_"$3,$4}' M1707.txt)
EOF

echo "Did not find $NOT_FOUND_N PDBs"
echo "Found $FOUND_N PDBs"
