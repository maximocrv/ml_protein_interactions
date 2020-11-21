#!/bin/python
import os
import multiprocessing as mp

# check if we are in a conda virtual env
try:
   os.environ["CONDA_DEFAULT_ENV"]
except KeyError:
   print("\tPlease init the conda environment!\n")
   exit()

import numpy as np
import pandas as pd

import parmed as pmd
import simtk.unit as su
import simtk.openmm as so
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile


DATA_PATH='../data/'

PDBS_PATH = DATA_PATH + 'pdbs/'
CLEANED_PDBS_PATH = DATA_PATH + 'pdbs_cleaned/'
PDBS_OPENMM = DATA_PATH + 'openmm/'

MUTATED_PDBS_PATH = DATA_PATH+'pdbs_mutated/'
CLEANED_MUTATED_PDBS_PATH = DATA_PATH+'pdbs_mutated_cleaned/'
MUTATED_PDBS_OPENMM = DATA_PATH + 'openmm_mutated/'

def generate_features(ids0, ids1, forcefield, system, param):
    """
    ids0: ids of the atoms for the 1st protein
    ids1: ids of the atoms for the 2nd protein
    """
    # sources
    # https://en.wikipedia.org/wiki/Electrostatics
    # https://en.wikipedia.org/wiki/Lennard-Jones_potential
    # https://en.wikipedia.org/wiki/Combining_rules
    
    # constants
    eps0 = 8.8541878128e-12 * su.farad * su.meter**-1
    e = 1.60217662e-19 * su.coulomb
    N = 6.02214179e23 * su.mole**-1 # Avogadro
    
    # scaling factors
    k0 = (N * (e*e) / (4.0 * np.pi * eps0))
    
    # get nonbonded interactions parameters for all atoms (Lennard-Jones and electrostatics)
    epsilon = np.array([a.epsilon for a in param.atoms])
    sigma = np.array([a.sigma for a in param.atoms])
    charge = np.array([a.charge for a in param.atoms])
    
    # pairwise epsilon with units
    E = np.sqrt(epsilon[ids0].reshape(-1,1) * epsilon[ids1].reshape(1,-1)) * param.atoms[0].uepsilon.unit
    
    # pairwise sigma with units
    S = 0.5*(sigma[ids0].reshape(-1,1) + sigma[ids1].reshape(1,-1)) * param.atoms[0].usigma.unit
    
    # pairwise partial charges
    Q = charge[ids0].reshape(-1,1) * charge[ids1].reshape(1,-1)

    # setup MD engine
    integrator = so.LangevinIntegrator(300*su.kelvin, 1/su.picosecond, 0.002*su.picoseconds)
    platform = so.Platform.getPlatformByName('CPU')
    simulation = so.app.Simulation(param.topology, system, integrator, platform)

    # set atom coordinates
    simulation.context.setPositions(param.get_coordinates()[0] * su.angstrom)

    # minimize energy
    simulation.minimizeEnergy()
    
    # get atom coordinates and compute distance matrix between subunits
    state = simulation.context.getState(getPositions=True)
    xyz = state.getPositions(asNumpy=True)
    D = np.linalg.norm(np.expand_dims(xyz[ids0], 1) - np.expand_dims(xyz[ids1], 0), axis=2) * su.angstrom
    
    # compute nonbonded potential energies
    U_LJ = (4.0 * E * (np.power(S/D, 12) - np.power(S/D, 6))).value_in_unit(su.kilojoule / su.mole)
    U_el = (k0 * Q / D).value_in_unit(su.kilojoule / su.mole)
    
    # debug print
    # print(f"U_LJ = {np.sum(U_LJ):.2f} kJ/mol; U_elec = {np.sum(U_el):.2f} kJ/mol")

    features = pd.DataFrame(data={'U_LJ':[U_LJ], 'U_el':[U_el], 'Dist_matrix':[D]})
    return features

def listdir_no_hidden():
    for f in os.listdir(MUTATED_PDBS_PATH):
        if not f.startswith('.'):
            yield MUTATED_PDBS_PATH, CLEANED_MUTATED_PDBS_PATH, MUTATED_PDBS_OPENMM, f
            return
    for f in os.listdir(PDBS_PATH):
        if not f.startswith('.'):
            yield PDBS_PATH, CLEANED_PDBS_PATH, PDBS_OPENMM, f

def pdb_parser(file):
    return [[char for char in subchain] for subchain  in file.split('_')[1:3]]

def pdb_clean_sim(args):
    orig_dir, clean_dir, sim_dir, fname = args
    # print(orig_dir, clean_dir, sim_dir, fname)
    # clean PDB
    if not os.path.exists(clean_dir + fname):
        pdb = pmd.load_file(orig_dir + fname)
        pdb.save(clean_dir + fname, overwrite=True)

        fixer = PDBFixer(filename=clean_dir + fname)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        # fixer.addSolvent(fixer.topology.getUnitCellDimensions())

        PDBFile.writeFile(fixer.topology, fixer.positions, \
                          open(clean_dir+fname, 'w'))
    else:
        fixer = PDBFixer(filename=clean_dir + fname)

    # Run simulation
    if not os.path.exists(sim_dir + fname):

        forcefield = so.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        system = forcefield.createSystem(fixer.topology, nonbondedMethod=so.app.NoCutoff)
        param = pmd.openmm.load_topology(fixer.topology, system=system, xyz=fixer.positions)

        # get indices of atoms for the 2 interacting subunits 
        # sub_unit_chains = pdb_parser(fname)
        sub_unit_chains = [['A'],['B']]
        # print(param.to_dataframe()['chain'])
        ids0, ids1 = (np.where(param.to_dataframe()['chain'].isin(cids))[0] for cids in sub_unit_chains)
        # print(sub_unit_chains,fname,ids0,ids1)

        features = generate_features(ids0, ids1, forcefield, system, param)

        print(f'done simulating: {fname}')
        # TODO: save features into directory
        features.to_csv(sim_dir + fname.split('.')[0] + '.csv')


if __name__ == '__main__':
    p = mp.Pool(5)
    for _ in p.imap_unordered(pdb_clean_sim, listdir_no_hidden()):
        pass
    print('finished')
