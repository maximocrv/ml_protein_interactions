#!/bin/python
"""
This script cleans and generates the features required for the models using OpenMM for simulation.
"""
import os
from pathlib import Path
from time import time
import multiprocessing as mp

import numpy as np
import parmed as pmd
import simtk.unit as su
import simtk.openmm as so
from pdbfixer import PDBFixer

from utilities import open_log
from constants import wt_features_path, wt_pdb_path, mut_features_path, mut_pdb_path


def generate_features(ids0, ids1, forcefield, system, param):
    """
    This function performs a minimization of the energy and computes the matrix features.

    :param ids0: ids of the atoms for the 1st protein
    :param ids1: ids of the atoms for the 2nd protein
    :param forcefield: forcefield for OpenMM simulation
    :param system: system for OpenMM simulation
    :param param: OpenMM parameters
    :return: features - to be used by ML models
    """

    # sources
    # https://en.wikipedia.org/wiki/Electrostatics
    # https://en.wikipedia.org/wiki/Lennard-Jones_potential
    # https://en.wikipedia.org/wiki/Combining_rules

    # constants
    eps0 = 8.8541878128e-12 * su.farad * su.meter**-1
    e = 1.60217662e-19 * su.coulomb
    N = 6.02214179e23 * su.mole**-1  # Avogadro

    # scaling factors
    k0 = (N * (e*e) / (4.0 * np.pi * eps0))

    # get nonbonded interactions parameters for all atoms
    # (Lennard-Jones and electrostatics)
    epsilon = np.array([a.epsilon for a in param.atoms])
    sigma = np.array([a.sigma for a in param.atoms])
    charge = np.array([a.charge for a in param.atoms])

    # pairwise epsilon with units
    E = np.sqrt(epsilon[ids0].reshape(-1, 1) * epsilon[ids1].reshape(1, -1)) * param.atoms[0].uepsilon.unit

    # pairwise sigma with units
    S = 0.5 * (sigma[ids0].reshape(-1, 1) + sigma[ids1].reshape(1, -1)) * param.atoms[0].usigma.unit

    # pairwise partial charges
    Q = charge[ids0].reshape(-1, 1) * charge[ids1].reshape(1, -1)

    # setup MD engine
    integrator = so.LangevinIntegrator(300*su.kelvin, 1/su.picosecond, 0.002*su.picoseconds)
    try:
        platform = so.Platform.getPlatformByName('CUDA')
    except Exception e:
        platform = so.Platform.getPlatformByName('CPU')

    simulation = so.app.Simulation(param.topology, system, integrator, platform)

    # set atom coordinates
    simulation.context.setPositions(param.get_coordinates()[0] * su.angstrom)

    # minimize energy
    simulation.minimizeEnergy()

    # get atom coordinates and compute distance matrix between subunits
    state = simulation.context.getState(getPositions=True)
    xyz = state.getPositions(asNumpy=True)
    D = np.linalg.norm(np.expand_dims(
        xyz[ids0], 1) - np.expand_dims(xyz[ids1], 0), axis=2) * su.angstrom

    # To choose the most relevant residues, we will first choose the pair of atoms with the lowest distance, and then
    # extract a submatrix around it. This way we preserve the chain order of the distance matrix.
    min_i = np.argmin(D)
    min_r, min_c = int(min_i/D.shape[1]), min_i % D.shape[1]

    # Number of interacting residues/particles considered relevant to be stored in the features
    n_interactions = 256

    ids0_min, ids0_max = min_r - n_interactions/2, min_r + n_interactions/2
    ids1_min, ids1_max = min_c - n_interactions/2, min_c + n_interactions/2

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

    # compute nonbonded potential energies
    U_LJ = (4.0 * E * (np.power(S/D, 12) - np.power(S/D, 6))).value_in_unit(su.kilojoule / su.mole)
    U_el = (k0 * Q / D).value_in_unit(su.kilojoule / su.mole)

    features = {'U_LJ': U_LJ, 'U_el': U_el, 'D_mat': D}

    return features


def listdir_no_hidden():
    """
    This function generates the PDB file names to be cleaned and simulated. Additionally, it creates the output
    directories if they are missing.

    :return: Iterator yielding file paths and pdb names
    """
    Path(mut_features_path).mkdir(parents=True, exist_ok=True)
    Path(wt_features_path).mkdir(parents=True, exist_ok=True)

    for f in Path(wt_pdb_path).glob('[!.]*.pdb'):
        yield wt_pdb_path, wt_features_path, f.name

    for f in Path(mut_pdb_path).glob('[!.]*.pdb'):
        yield mut_pdb_path, mut_features_path, f.name


def pdb_parser(file):
    """
    This function parses through a PDB file.

    :param file: Input file pertaining to pdb
    :return: Parsed key words from file name
    """
    return [[char for char in subchain] for subchain in file.split('_')[1:3]]


def pdb_clean_sim(args):
    """
    Top-level function to be executed in parallel to clean and generate features.

    :param args: Input and output directories, pdb name.
    :return:
    """
    input_dir, output_dir, fname = args
    # print(input_dir, output_dir, fname)
    if not Path(output_dir + fname).exists():
        # clean PDB
        pdb = pmd.load_file(input_dir + fname)
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
            ids0, ids1 = (np.where(param.to_dataframe()['chain'].isin(cids))[
                          0] for cids in sub_unit_chains)
            # print(sub_unit_chains,fname,ids0,ids1)

            features = generate_features(ids0, ids1, forcefield, system, param)

            print(f'done simulating: {fname}')

            # stack 3 matrices into 1
            combined_mat = np.stack((features["U_LJ"], features["U_el"], features["D_mat"]))

            np.save(output_dir + '/' + basename + '.npy', combined_mat)

            print(f'saved features: {fname}')

        except Exception as e:
            print(f'could not simulate: {fname} Exception: {e}')
            return 1, f'E;{fname};{e}'

    return 0, f'S;{fname};'


if __name__ == '__main__':
    # check if we are in a conda virtual env
    try:
        os.environ["CONDA_DEFAULT_ENV"]
    except KeyError:
        print("\tPlease init the conda environment!\n")
        exit(1)

    log = open_log('pdb_pipeline')

    log.write('# E: Error    S: Saved\n')
    log.write('# status    name    exception\n')

    # no. of PDBs that could not be simulated
    n_unsimulatables = 0
    n_total = 0
    # start multiprocessing of the simulations
    start_t = time()
    p = mp.Pool(5)
    for unsim, msg in p.imap_unordered(pdb_clean_sim, listdir_no_hidden()):
        n_total += 1
        n_unsimulatables += unsim
        log.write(msg + '\n')

    # output
    exec_t = time() - start_t

    print(f'finished in {exec_t}s could not simulate {n_unsimulatables} PDBs')
    log.write(f'# parsed a total of {n_total} PDBs\n')
    log.write(
        f'# finished in {exec_t}s could not simulate {n_unsimulatables} PDBs\n')

    log.close()
else:
    raise Exception('\tPlease execute this script directly.\n')
