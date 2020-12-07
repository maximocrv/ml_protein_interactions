"""This script contains utility functions used throughout the rest of the codebase."""

import numpy as np
import pandas as pd

from pathlib import Path
import time


def load_data():
    df = pd.read_csv('../data/mlp_features.csv')
    df.drop(columns='mut', inplace=True)
    df = (df - df.mean(axis=0)) / df.std(axis=0)
    # print(df.head())
    return np.array(df.iloc[:, 1:-1], dtype=np.float32), np.array(df.iloc[:, -1], dtype=np.float32)


def load_data_features():
    # function we will use to load in data for both xgb and mlp
    pass


def load_data_mats():
    # function we will use to load in data for hydranet
    pass


def feature_expansion():
    # feature expansion for xgboost model
    pass


def data_preprocessor():
    # function for log transforms/ normalizing etc
    pass


def open_log(name):
    """Open a file with the current time attached to its filename.
    this file will be created in a folder with the name provided
    in the argument.
    """
    out_dir = f'log-{name}'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    time_str = time.strftime('%d-%m-%yT%H:%M:%S')
    return open(f'{out_dir}/{name}.{time_str}.out', 'w')

# we have too many functions that have hard coded file paths etc, would be better to generalize them and add arguments
# so that we can reuse them in other parts of the code
