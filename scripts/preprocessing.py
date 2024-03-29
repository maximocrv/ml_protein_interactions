#!/bin/python
"""
This script preprocesses the generated features to create a CSV file to be used by the MLP and XGBoost models.
"""
import os
import re
import math
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd

from constants import skempi_csv, wt_features_path, mut_features_path, mlp_features, R
from utilities import open_log


def preprocessing(pandas_row):
    """
    This function generates the features that will be used by the MLP and XGboost given the SKEMPI v2 datapoint (the
    wild-type and mutant complexes).

    The feature matrices should have been generated beforehand using `pdb_pipeline.py`.

    :param pandas_row: Row of the pandas dataframe which contains all relevant information in order to load specific
    mutants and wild types.
    :return: Dictionary containing means and standard deviations of all channels of both the mutants and the wild types,
    temperature, and the corresponding DDG. Also contains the name of the mutation which by default has the wild type
    name stored within it.
    """
    name_wt = pandas_row[1].iloc[0]
    name_mut = pandas_row[1].iloc[0] + '_' + \
        pandas_row[1].iloc[2].replace(',', '_')

    if not Path(mut_features_path + name_mut + '.npy').exists():
        print(f'ERROR: {name_mut} does not exist.')
        return None
    if not Path(wt_features_path + name_wt + '.npy').exists():
        print(f'ERROR: {name_wt} does not exist.')
        return None

    wt_tensor = np.load(wt_features_path + name_wt + '.npy')
    mut_tensor = np.load(mut_features_path + name_mut + '.npy')
    u_lj_wt = wt_tensor[0, :, :]
    u_lj_mut = mut_tensor[0, :, :]
    u_el_wt = wt_tensor[1, :, :]
    u_el_mut = mut_tensor[1, :, :]
    d_mat_wt = wt_tensor[2, :, :]
    d_mat_mut = mut_tensor[2, :, :]

    d_mat_wt_mean, d_mat_wt_std = get_mean_std(d_mat_wt)
    u_lj_wt_mean, u_lj_wt_std = get_mean_std(u_lj_wt)
    u_el_wt_mean, u_el_wt_std = get_mean_std(u_el_wt)

    d_mat_mut_mean, d_mat_mut_std = get_mean_std(d_mat_mut)
    u_lj_mut_mean, u_lj_mut_std = get_mean_std(u_lj_mut)
    u_el_mut_mean, u_el_mut_std = get_mean_std(u_el_mut)
    temp = float(re.match("[0-9]*", pandas_row[1]['Temperature'])[0])
    if math.isnan(temp):
        raise ValueError('temperature should not be NaN.')

    # calculate DDG
    A_wt = pandas_row[1]['Affinity_wt_parsed']
    A_mut = pandas_row[1]['Affinity_mut_parsed']

    DG_wt = R * temp * np.log(A_wt)
    DG_mut = R * temp * np.log(A_mut)
    DDG = DG_mut - DG_wt

    # debug print
    print(f'parsed {name_mut}')

    return {"mut": name_mut,
            "d_mat_wt_mean": d_mat_wt_mean, "d_mat_wt_std": d_mat_wt_std,
            "u_lj_wt_mean": u_lj_wt_mean, "u_lj_wt_std": u_lj_wt_std,
            "u_el_wt_mean": u_el_wt_mean, "u_el_wt_std": u_el_wt_std,
            "d_mat_mut_mean": d_mat_mut_mean, "d_mat_mut_std": d_mat_mut_std,
            "u_lj_mut_mean": u_lj_mut_mean, "u_lj_mut_std": u_lj_mut_std,
            "u_el_mut_mean": u_el_mut_mean, "u_el_mut_std": u_el_mut_std,
            "temp": temp, "DDG": DDG}


def get_mean_std(mat):
    """
    This function returns the overall mean of the matrix,
    and the averaged STD of each row.

    :param mat: Input matrix
    :return: Overall mean and average standard deviation of each row
    """
    return np.mean(mat), np.mean(np.std(mat, axis=1))


if __name__ == '__main__':
    # check if we are in a conda virtual env
    try:
        os.environ["CONDA_DEFAULT_ENV"]
    except KeyError:
        print("\tPlease init the conda environment!\n")
        exit(1)

    log = open_log('preprocessing_mlp')

    df = pd.read_csv(skempi_csv, sep=';')

    # filter duplicated
    df = df[~df.duplicated(subset=["#Pdb", "Mutation(s)_cleaned"])]

    # remove without target
    df = df.dropna(subset=['Affinity_mut_parsed'])
    df = df.dropna(subset=['Affinity_wt_parsed'])
    df = df.dropna(subset=['Temperature'])

    log.write('\tloaded SKEMPI dataset:\n')
    log.write(str(df.head()))
    log.write('\n')

    df_out = pd.DataFrame(columns=[
        "mut", "d_mat_wt_mean", "d_mat_wt_std",
        "u_lj_wt_mean",  "u_lj_wt_std", "u_el_wt_mean",  "u_el_wt_std",
        "d_mat_mut_mean", "d_mat_mut_std", "u_lj_mut_mean",  "u_lj_mut_std",
        "u_el_mut_mean",  "u_el_mut_std", "temp", "DDG"
    ])

    n_non_existant = 0

    for data in mp.Pool(5).imap_unordered(preprocessing, df.iterrows()):
        if data is None:
            n_non_existant += 1
        else:
            df_out = df_out.append(data, ignore_index=True)

    print(f'{n_non_existant} PDBs do not have features.')
    print(str(df_out.head()))

    df_out.to_csv(mlp_features)

    log.write(f'\t{n_non_existant} PDBs do not have features.\n')
    log.write('\tMLP features:\n')
    log.write(str(df_out.head()))

    log.close()
else:
    raise Exception('\tPlease execute this script directly.\n')
