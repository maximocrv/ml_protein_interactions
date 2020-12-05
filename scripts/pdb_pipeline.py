#!/bin/python
# Run this script like:
# script -q -c 'python pdb_pipeline.py' /dev/null | tee pdb_pipeline.out

# TODO: replace os with pathlib
import os
from pathlib import Path
from time import time
import multiprocessing as mp

# check if we are in a conda virtual env
try:
   os.environ["CONDA_DEFAULT_ENV"]
except KeyError:
   print("\tPlease init the conda environment!\n")
   exit(1)

import numpy as np
# import pandas as pd

import parmed as pmd
import simtk.unit as su
import simtk.openmm as so
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile


DATA_PATH='../data/'

PDBS_PATH = DATA_PATH + 'pdbs_wt/'
PDBS_OPENMM = DATA_PATH + 'openmm/'

MUTATED_PDBS_PATH = DATA_PATH+'pdbs_mutated/'
MUTATED_PDBS_OPENMM = DATA_PATH + 'openmm_mutated/'

# Number of interacting residues/particles considered
# relevant to be stored in the features
n_interactions = 256
# max distance for two atoms to be considered 'interacting'. TODO: implement this
inter_dist = 5 * su.angstrom

def generate_features(ids0, ids1, forcefield, system, param):
    """
    ids0: ids of the atoms for the 1st protein
    ids1: ids of the atoms for the 2nd protein

    IMPORTANT! `features` should be a dictionnary whose keys are the same as the
                name of the folders being created in the simulation output.
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
    platform = so.Platform.getPlatformByName('CUDA')
    simulation = so.app.Simulation(param.topology, system, integrator, platform)

    # set atom coordinates
    simulation.context.setPositions(param.get_coordinates()[0] * su.angstrom)

    # minimize energy
    simulation.minimizeEnergy()
    
    # get atom coordinates and compute distance matrix between subunits
    state = simulation.context.getState(getPositions=True)
    xyz = state.getPositions(asNumpy=True)
    D = np.linalg.norm(np.expand_dims(xyz[ids0], 1) - np.expand_dims(xyz[ids1], 0), axis=2) * su.angstrom

    # print(D.shape, ids1.shape, ids0.shape)

    # to choose the most relevant residues, we will first choose the pair of atoms with the
    # lowest distance, and then extract a submatrix around it.
    # This way we preserve the chain order of the distance matrix.
    min_i = np.argmin(D)
    min_r, min_c = int(min_i/D.shape[1]), min_i%D.shape[1]
    # print(f'minimum distance: {np.min(D)}    position: {(min_r, min_c)}')
    
    ids0_min, ids0_max = min_r-n_interactions/2, min_r+n_interactions/2
    ids1_min, ids1_max = min_c-n_interactions/2, min_c+n_interactions/2
    
    if ids0_min < 0:
        ids0_max -= ids0_min
        ids0_min = 0
    elif ids0_max >= D.shape[0]:
        ids0_min -= ids0_max - D.shape[0] + 1
        ids0_max = D.shape[0]-1
        
    if ids1_min < 0:
        ids1_max -= ids1_min
        ids1_min = 0
    elif ids1_max >= D.shape[1]:
        ids1_min -= ids1_max - D.shape[1] + 1
        ids1_max = D.shape[1]-1
    
    ids0_interacting = np.arange(ids0_min, ids0_max, dtype=np.int32)
    ids1_interacting = np.arange(ids1_min, ids1_max, dtype=np.int32)

    D = D[np.ix_(ids0_interacting, ids1_interacting)]
    S = S[np.ix_(ids0_interacting, ids1_interacting)]
    Q = Q[np.ix_(ids0_interacting, ids1_interacting)]
    E = E[np.ix_(ids0_interacting, ids1_interacting)]

    # print(D.shape, S.shape, Q.shape, E.shape)
    
    # compute nonbonded potential energies
    U_LJ = (4.0 * E * (np.power(S/D, 12) - np.power(S/D, 6))).value_in_unit(su.kilojoule / su.mole)
    U_el = (k0 * Q / D).value_in_unit(su.kilojoule / su.mole)
    
    # print(U_LJ.shape, U_el.shape)
    
    # debug print
    # print(f"U_LJ = {np.sum(U_LJ):.2f} kJ/mol; U_elec = {np.sum(U_el):.2f} kJ/mol")
    # print(U_LJ.shape, U_el.shape, D.shape)

    features = {'U_LJ':U_LJ, 'U_el':U_el, 'D_mat':D}
    return features

def listdir_no_hidden():
    """Generates the PDB file names to be cleaned and simulated.

    Additionally, it creates the output directories if they are missing.
    IMPORTANT! The name of the subfolders in the simulation output dir should
                match the name of the keys in the `features` dictionnary. 
    """
    Path(MUTATED_PDBS_OPENMM).mkdir(parents=True, exist_ok=True)
    Path(MUTATED_PDBS_OPENMM+'U_LJ/').mkdir(parents=True, exist_ok=True)
    Path(MUTATED_PDBS_OPENMM+'U_el/').mkdir(parents=True, exist_ok=True)
    Path(MUTATED_PDBS_OPENMM+'D_mat/').mkdir(parents=True, exist_ok=True)

    Path(PDBS_OPENMM).mkdir(parents=True, exist_ok=True)
    Path(PDBS_OPENMM+'U_LJ/').mkdir(parents=True, exist_ok=True)
    Path(PDBS_OPENMM+'U_el/').mkdir(parents=True, exist_ok=True)
    Path(PDBS_OPENMM+'D_mat/').mkdir(parents=True, exist_ok=True)

    for f in Path(PDBS_PATH).glob('[!.]*.pdb'):
        yield PDBS_PATH, PDBS_OPENMM, f.name

    for f in Path(MUTATED_PDBS_PATH).glob('[!.]*.pdb'):
        yield MUTATED_PDBS_PATH, MUTATED_PDBS_OPENMM, f.name

def pdb_parser(file):
    return [[char for char in subchain] for subchain  in file.split('_')[1:3]]

def pdb_clean_sim(args):
    """ Main function to be executed in parallel.
    """
    orig_dir, sim_dir, fname = args
    # print(orig_dir, sim_dir, fname)
    if not Path(sim_dir + fname).exists():
        # clean PDB
        pdb = pmd.load_file(orig_dir + fname)
        pdb.save('/tmp/' + fname, overwrite=True)
    
        fixer = PDBFixer(filename='/tmp/' + fname)
        Path('/tmp/' + fname).unlink()
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        # print(f'number of non-standard residues in {fname}: {len(fixer.nonstandardResidues)}')
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
    
        # fixer.addSolvent(fixer.topology.getUnitCellDimensions())
    
        # Run simulation
        try:
            forcefield = so.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
            system = forcefield.createSystem(fixer.topology, nonbondedMethod=so.app.NoCutoff)
            param = pmd.openmm.load_topology(fixer.topology, system=system, xyz=fixer.positions)
    
            basename = '.'.join(fname.split('.')[:-1])
    
            # get indices of atoms for the 2 interacting subunits 
            sub_unit_chains = pdb_parser(basename)
            # print(param.to_dataframe()['chain'])
            ids0, ids1 = (np.where(param.to_dataframe()['chain'].isin(cids))[0] for cids in sub_unit_chains)
            # print(sub_unit_chains,fname,ids0,ids1)
    
            features = generate_features(ids0, ids1, forcefield, system, param)
    
            print(f'done simulating: {fname}')
    
            for feature in features: 
                # print('saving to: '+sim_dir + feature + '/' + basename + '.csv')
                # print(feature, features[feature].shape)
                # np.savetxt(sim_dir + feature + '/' + basename + '.csv', features[feature], delimiter=',')
                np.save(sim_dir + feature + '/' + basename + '.npy', features[feature])
    
            print(f'saved features: {fname}')

        except Exception as e:
            print(f'could not simulate: {fname} Exception: {e}')
            return 1

    return 0


if __name__ == '__main__':
    start_t = time()
    # no. of PDBs that could not be simulated
    n_unsimulatables = 0
    p = mp.Pool(5)
    for unsim in p.imap_unordered(pdb_clean_sim, listdir_no_hidden()):
        n_unsimulatables += unsim
    exec_t = time() - start_t
    print(f'finished in {exec_t}s could not simulate {n_unsimulatables} PDBs')
