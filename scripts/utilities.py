"""
This script contains utility functions used throughout the rest of the codebase.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from constants import mlp_features


def load_data():
    """
    Load dataset to use for MLP and XGboost

    :return: Input feature array and targets
    """
    df = pd.read_csv(mlp_features)
    df.drop(columns='mut', inplace=True)
    # df = (df - df.mean(axis=0)) / df.std(axis=0)
    # print(df.head())
    return np.array(df.iloc[:, 1:-1], dtype=np.float32), np.array(df.iloc[:, -1], dtype=np.float32)


def open_log(name):
    """
    Open a file with the current time attached to its filename. This file will be created in a folder with the name
    provided in the argument.

    :param name: Name of the logging directory (will create a new one if it does not already exist).
    :return: Logger under your chosen name
    """
    out_dir = f'log-{name}'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    time_str = time.strftime('%d-%m-%yT%H:%M:%S')
    return open(f'{out_dir}/{name}.{time_str}.out', 'w')


def clip_features_inplace(feature, low_fraction=0.1, up_fraction=None):
    """
    Clips the values in the feature vector by setting the lower sorted `fraction` of elements to be equal to the value
    corresponding to the position at `len(feature)*fraction`.

    Similarly for the upper sorted `fraction` of elements which take the value of the element corresponding to the
    position `len(feature)*(1-fraction)`.

    :param feature: Input feature
    :param low_fraction: Used to determine lower clipping bound
    :param up_fraction: Used to determine upper clipping bound
    :return:  Returns the modified-inplace `feature` vector.
    """
    if up_fraction is None:
        up_fraction = low_fraction
    N = len(feature)-1
    lower_i = int(N*low_fraction)
    upper_i = int(N*(1-up_fraction))
    sorted_vals = np.sort(feature)
    feature[feature < sorted_vals[lower_i]] = sorted_vals[lower_i]
    feature[feature > sorted_vals[upper_i]] = sorted_vals[upper_i]
    
    return feature


def build_poly(x, degree):
    """
    Builds polynomial basis function (for both input vectors or input arrays).

    :param x: Input features.
    :param degree: Degree of the polynomial basis.
    :return: Horizontally concatenated array containing all the degree bases up to and including the selected degree
    parameter.
    """
    if degree == 0:
        x = np.ones((x.shape[0], 1))
    else:
        x = np.repeat(x[..., np.newaxis], degree, axis=-1)
        x = x ** np.arange(1, degree + 1)
        x = np.concatenate(x.transpose(2, 0, 1), axis=-1)
    return x


def cross_channel_features(x):
    """
    Generate matrix containing product of all features with one another (except themselves).

    :param x: Input features.
    :return: Numpy array containing product of all channels with each other.
    """
    cross_x = np.zeros((x.shape[0], np.sum(np.arange(x.shape[1]))))

    count = 0
    for i in range(x.shape[1]):
        for j in range(i+1, x.shape[1]):
            cross_x[:, count] = x[:, i] * x[:, j]
            count += 1

    return cross_x


def transform_data(x_tr, x_te, degree, cross=True, log=True):
    """
    Performs the data transformation and feature expansion on the input features. Concatenates the polynomial expansion
    basis, logarithmic basis (of positive columns), cross channel correlations, and the intercept term for the training
    and testing data.

    :param x_tr: Train input features.
    :param x_te: Test input features.
    :param degree: Degree of the polynomial basis.
    :param cross: Boolean to decide if you want to include the the cross channel correlations.
    :param log: Boolean to decide if you want to include the log of the (positive) columns.
    :return: Transformed, horizontally concatenated input feature matrix.
    """
    if cross:
        x_tr_cross = cross_channel_features(x_tr)
        x_te_cross = cross_channel_features(x_te)

    if log:
        neg_cols_te = np.any(x_tr <= 0, axis=0)
        neg_cols_tr = np.any(x_te <= 0, axis=0)
        neg_cols = np.logical_or(neg_cols_te, neg_cols_tr)

        x_tr_log = np.log(x_tr[:, ~neg_cols])
        x_te_log = np.log(x_te[:, ~neg_cols])

    x_tr = build_poly(x_tr, degree)
    x_te = build_poly(x_te, degree)

    if cross and log:
        x_tr = np.concatenate((x_tr, x_tr_cross, x_tr_log), axis=1)
        x_te = np.concatenate((x_te, x_te_cross, x_te_log), axis=1)
    if cross and not log:
        x_tr = np.concatenate((x_tr, x_tr_cross), axis=1)
        x_te = np.concatenate((x_te, x_te_cross), axis=1)
    if not cross and log:
        x_tr = np.concatenate((x_tr, x_tr_log), axis=1)
        x_te = np.concatenate((x_te, x_te_log), axis=1)

    x_tr, tr_mean, tr_sd = standardize_data(x_tr)

    x_te[:, tr_sd > 0] = (x_te[:, tr_sd > 0] - tr_mean[tr_sd > 0]) / tr_sd[tr_sd > 0]
    x_te[:, tr_sd == 0] = x_te[:, tr_sd == 0] - tr_mean[tr_sd == 0]

    x_tr = np.concatenate((np.ones((x_tr.shape[0], 1)), x_tr), axis=1)
    x_te = np.concatenate((np.ones((x_te.shape[0], 1)), x_te), axis=1)

    return x_tr, x_te


def standardize_data(x):
    """
    Standardization of the dataset, so that mean = 0 and std = 1. The data is filtered such that all columns where the
    standard deviation is equal to zero simply have their mean subtracted, in order to avoid division by zero errors.

    :param x: Input dataset.
    :return: Standardized dataset.
    """
    col_means = np.nanmean(x, axis=0)
    col_sd = np.nanstd(x, axis=0)

    x[:, col_sd > 0] = (x[:, col_sd > 0] - col_means[col_sd > 0]) / col_sd[col_sd > 0]
    x[:, col_sd == 0] = x[:, col_sd == 0] - col_means[col_sd == 0]

    return x, col_means, col_sd


def generate_scatter_plot(target, pred):
    """
    Generates a scatter plot between the target and predicted values with annotated Pearson R and RMSE scores.\

    :param target: Label data
    :param pred: Predicted data
    :return: Generates a scatter plot
    """
    fig = plt.figure()
    
    plt.scatter(pred, target, s=10)
    plt.plot(np.arange(np.min(pred), np.max(pred)), np.arange(np.min(pred), np.max(pred)), 'r')
    
    plt.title('Predicted $\Delta\Delta$G vs Target $\Delta \Delta$G')
    plt.xlabel('Predicted $\Delta\Delta$G')
    plt.ylabel('Target $\Delta \Delta$G')
    
    plt.set_xlim = (np.min(pred), np.max(pred)) 
    plt.tight_layout()
    
    R = pearsonr(target, pred)[0]
    rmse_te = np.sqrt(np.mean((pred - target) ** 2))
    
    plt.annotate(f"Pearson's R: {R:.4f}", (4, -5))
    plt.annotate(f"Test RMSE: {rmse_te:.4f}", (4, -7))
