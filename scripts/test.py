import os
from sys import stdout

import pandas as pd

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

root_dir_pdbs = '../data/pdbs_mutated_cleaned/'
root_dir_openmm = '../data/openmm_output/'
pdb_list = os.listdir(root_dir_pdbs)

unsimulatables = []

for ele in pdb_list:
    if os.path.exists(root_dir_openmm + f"{ele[:-4]}"):
        print(f"{ele[:-4]}")

                                
