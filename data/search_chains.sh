#!/bin/bash

for f in pdbs_mutated/*.pdb; do
	chain1=$(echo "$f" | cut -d'_' -f3)
	chain2=$(echo "$f" | cut -d'_' -f4)
	if [ $(echo -n $chain1 | wc -c) -gt 1 ] || [ $(echo -n $chain2 | wc -c) -gt 1 ]; then
		s="$(cat "$f" | tr -s ' ' | cut -d' ' -f5 | sort | uniq | tail -n3 | xargs echo -n)"
		if [ "$s" = "A B END" ]; then
			echo "$chain1" "$chain2"
			echo "PROBLEM IN FILE $f"
		fi
	fi
done
