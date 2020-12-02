#!/bin/python
# Run this script like:
# script -q -c 'python preprocessing_mlp.py' /dev/null | tee preprocessing_mlp.out

import os
from pathlib import Path

# check if we are in a conda virtual env
try:
   os.environ["CONDA_DEFAULT_ENV"]
except KeyError:
   print("\tPlease init the conda environment!\n")
   exit(1)

import numpy as np
import multiprocessing as mp

import pandas as pd

import math
import re

DATA_PATH='../data/'
SKEMPI_CSV=DATA_PATH + 'skempi_v2_cleaned.csv'
WT_FEATURE_PATH=DATA_PATH + 'openmm/'
MUT_FEATURE_PATH=DATA_PATH + 'openmm_mutated/'
MLP_OUTPUT_PATH=DATA_PATH + 'mlp_features.csv'

R = (8.314/4184)  # kcal mol^-1 K^-1

def get_mean_std(mat):
    return np.mean(mat), np.mean(np.std(mat,axis=1))

def MLP_preprocessing(pandas_row):
    name_wt  = pandas_row[1].iloc[0]
    name_mut = pandas_row[1].iloc[0] + '_' + pandas_row[1].iloc[2].replace(',', '_')

    # matrix_features = ['D_mat', 'U_LJ', 'U_el']
    # for feature in matrix_features:
    #     pass

    if not Path(MUT_FEATURE_PATH+'D_mat/'+name_mut+'.npy').exists():
        print(f'ERROR: {name_mut} does not exist.')
        return None
    if not Path(WT_FEATURE_PATH+'D_mat/'+name_wt+'.npy').exists():
        print(f'ERROR: {name_wt} does not exist.')
        return None

    d_mat_wt  = np.load(WT_FEATURE_PATH +'D_mat/'+name_wt +'.npy')
    d_mat_mut = np.load(MUT_FEATURE_PATH+'D_mat/'+name_mut+'.npy')
    u_lj_wt   = np.load(WT_FEATURE_PATH +'U_LJ/'+name_wt +'.npy')
    u_lj_mut  = np.load(MUT_FEATURE_PATH+'U_LJ/'+name_mut+'.npy')
    u_el_wt   = np.load(WT_FEATURE_PATH +'U_el/'+name_wt +'.npy')
    u_el_mut  = np.load(MUT_FEATURE_PATH+'U_el/'+name_mut+'.npy')

    d_mat_wt_mean, d_mat_wt_std = get_mean_std(d_mat_wt)
    u_lj_wt_mean,  u_lj_wt_std  = get_mean_std(u_lj_wt)
    u_el_wt_mean,  u_el_wt_std  = get_mean_std(u_el_wt)
    
    d_mat_mut_mean, d_mat_mut_std = get_mean_std(d_mat_mut)
    u_lj_mut_mean,  u_lj_mut_std  = get_mean_std(u_lj_mut)
    u_el_mut_mean,  u_el_mut_std  = get_mean_std(u_el_mut)
    temp = float(re.match("[0-9]*", pandas_row[1]['Temperature'])[0])
    if math.isnan(temp):
        raise ValueError('temperature should not be NaN.')
    
    # calculate DDG
    A_wt  = pandas_row[1]['Affinity_wt_parsed']
    A_mut = pandas_row[1]['Affinity_mut_parsed']

    DG_wt = R * temp * np.log(A_wt)
    DG_mut = R * temp * np.log(A_mut)
    DDG = DG_mut - DG_wt

    # debug print
    print(f'parsed {name_mut}')

    return {"mut": name_mut, "d_mat_wt_mean":d_mat_wt_mean, "d_mat_wt_std":d_mat_wt_std, "u_lj_wt_mean":u_lj_wt_mean,  "u_lj_wt_std":u_lj_wt_std, "u_el_wt_mean":u_el_wt_mean,  "u_el_wt_std":u_el_wt_std, "d_mat_mut_mean":d_mat_mut_mean, "d_mat_mut_std":d_mat_mut_std, "u_lj_mut_mean":u_lj_mut_mean,  "u_lj_mut_std":u_lj_mut_std, "u_el_mut_mean":u_el_mut_mean,  "u_el_mut_std":u_el_mut_std, "temp":temp, "DDG": DDG}


if __name__ == '__main__':
    df = pd.read_csv(SKEMPI_CSV, sep=';')
    
    # filter duplicated
    df = df[~df.duplicated(subset=["#Pdb", "Mutation(s)_cleaned"])]

    # remove without target
    df = df.dropna(subset=['Affinity_mut_parsed'])
    df = df.dropna(subset=['Affinity_wt_parsed'])
    df = df.dropna(subset=['Temperature'])

    df_out = pd.DataFrame(columns=["mut", "d_mat_wt_mean", "d_mat_wt_std", "u_lj_wt_mean",  "u_lj_wt_std", "u_el_wt_mean",  "u_el_wt_std", "d_mat_mut_mean", "d_mat_mut_std", "u_lj_mut_mean",  "u_lj_mut_std", "u_el_mut_mean",  "u_el_mut_std", "temp", "DDG"])
    n_non_existant = 0
    for data in mp.Pool(5).imap_unordered(MLP_preprocessing, df.iterrows()):
        if data is None:
            n_non_existant += 1
        else:
            df_out = df_out.append(data, ignore_index=True)

    print(f'{n_non_existant} PDBs do not have features.')
    print(df_out)
    df_out.to_csv(MLP_OUTPUT_PATH)

