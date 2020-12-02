import sys

with open('pdb_pipeline.out') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
        cheese = line.split()[0:2]
        if cheese == ['could', 'not']:
            count += 1

print(count)
