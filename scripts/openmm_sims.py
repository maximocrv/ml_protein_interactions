import os
from sys import stdout

import pandas as pd

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

root_dir_pdbs = '../data/pdbs_mutated_cleaned/'
root_dir_openmm = '../data/openmm_output/'

def listdir_no_hidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

pdb_list = listdir_no_hidden(root_dir_pdbs)

unsimulatables = []

for ele in pdb_list:
    if not os.path.exists(root_dir_openmm + ele[:-4]):
        try:
            pdb = PDBFile(root_dir_pdbs + ele)
            forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
            # changed nonbondedMethod to NoCutoff from PME
            system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                                             nonbondedCutoff=1*nanometer, constraints=HBonds)
            integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
            simulation = Simulation(pdb.topology, system, integrator)
            simulation.context.setPositions(pdb.positions)
            simulation.minimizeEnergy()
            #simulation.reporters.append(PDBReporter(root_dir_openmm + f'{ele[:-4]}', 1000))
            simulation.reporters.append(StateDataReporter(root_dir_openmm + ele[:-4], 1000, step=True, time=True,
                                                          potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                                          temperature=True, volume=True, density=True))
            simulation.step(20000)

        except:
            print("could not simulate " +  ele)
            unsimulatables.append(ele)

unsimulatables = pd.DataFrame(unsimulatables)
unsimulatables.to_csv('unsimulatables.csv')
