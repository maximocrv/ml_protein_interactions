"""This script contains functions used for preprocessing the data."""
import numpy as np


def set_nan(x):
    """
    Converts all -999 entries to nans.

    :param x: Input dataset.
    :return: Input dataset with all -999's replaced by nans.
    """
    x[x == -999] = np.nan
    return x


def remove_constant_columns(x):
    """
    Removes columns containing one single value (be it np.nan or other values).

    :param x: Input feature data.
    :return: Cleaned input data excluding columns containing one element.
    """
    x = set_nan(x)

    nan_count = np.sum(np.isnan(x), axis=0)
    only_nans = np.where(nan_count == x.shape[0])
    x = np.delete(x, only_nans, axis=1)

    single_list = []
    for i in range(x.shape[1]):
        nan_rows = np.isnan(x[:, i])
        unique, counts = np.unique(x[~nan_rows, i], return_counts=True, axis=0)
        if len(unique) == 1:
            single_list.append(i)

    x = np.delete(x, single_list, axis=1)

    return x


def convert_nan(x, nan_mode='mode'):
    """
    Replace all -999 entries by the mean, median, or mode of their respective columns.

    :param nan_mode: Mean, median or mode.
    :param x: Input data.
    :return: Input data containing column means in place of -999 entries.
    """
    if nan_mode == 'mean':
        col_vals = np.nanmean(x, axis=0)

    elif nan_mode == 'median':
        col_vals = np.nanmedian(x, axis=0)

    elif nan_mode == 'mode':
        col_vals = np.zeros((1, x.shape[1]))

        for i in range(x.shape[1]):
            nan_rows = np.isnan(x[:, i])
            unique, counts = np.unique(x[~nan_rows, i], return_counts=True, axis=0)
            col_vals[:, i] = unique[counts.argmax()]

    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_vals, inds[1])
    return x


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


def balance_all(y, x):
    """
    Balances the datasets by selecting and cutting a random subsets of misses, which are more abundant,
    to obtain an equal number of hits and misses.

    :param y: Label data.
    :param x: Input features.
    :return: Balanced dataset containing a randomly selected 50/50 distribution of signals and background noise.
    """
    x = set_nan(x)

    hits = np.sum(y[y == 1])
    misses = - np.sum(y[y == -1])

    diff = misses - hits
    allmiss_indexes = np.argwhere(y == -1)
    cut_indexes = np.random.choice(allmiss_indexes.flatten(), size=np.int(diff), replace=False)
    xv = np.delete(x, cut_indexes, axis=0)
    yv = np.delete(y, cut_indexes, axis=0)

    return yv, xv


def balance_fromnans(y, x):
    """
    Balances the datasets by preferably cutting features with nans. To be used with the entire dataset and
    not with spit-number-of-jets specific subdatasets.

    :param y: Label data.
    :param x: Input features.
    :return: yv and xv, with equal hits and misses
    """
    x = set_nan(x)

    hits = np.sum(y[y == 1])
    misses = len(y[y == 0])
    diff = misses - hits

    features = np.array([23, 24, 25, 4, 5, 6, 12, 26, 27, 28])
    nancount = np.isnan(x[:, features])

    nancount_allfeat = (np.sum(nancount, 1) == features.shape[0]) & (y == 0)
    misses_subgroup = np.sum(nancount_allfeat)

    if misses_subgroup < diff:
        # not enough misses to cut that are nans in the selected features
        amount = misses_subgroup
    else:
        amount = diff

    all_indexes = np.argwhere(nancount_allfeat)
    cut_indexes = np.random.choice(all_indexes.flatten(), size=np.int(amount), replace=False)

    xv = np.delete(x, cut_indexes, axis=0)
    yv = np.delete(y, cut_indexes, axis=0)

    return yv, xv


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


def split_data(x, y, ratio, seed=1):
    """
    Split the dataset into training and testing data.

    :param x: Input features.
    :param y: Label data.
    :param ratio: Ratio of training data.
    :param seed: Seed for random permutations (to allow for reproducibility).
    :return: Training and testing sets for both the input features and label data.
    """
    np.random.seed(seed)

    split = int(ratio * x.shape[0])
    train_ind = np.random.permutation(np.arange(x.shape[0]))[:split]
    test_ind = np.random.permutation(np.arange(x.shape[0]))[split:]

    x_tr, y_tr = x[train_ind], y[train_ind]
    x_te, y_te = x[test_ind], y[test_ind]

    return x_tr, y_tr, x_te, y_te


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    :param y: Label data.
    :param tx: Input features.
    :param batch_size: Size of the batch to generate (default is 1).
    :param num_batches: Number of batches to generate for iterator.
    :param shuffle: Boolean to decide whether to shuffle data or not (default is True).
    :return: Iterator containing the requested number of batches with the given batch size.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_k_indices(y, k_fold, seed):
    """
    Build the set of k indices for k-fold validation.

    :param y: Label data.
    :param k_fold: Number of folds (i.e. dataset partitions).
    :param seed: Seed for random permutations (to allow for reproducibility).
    :return: Array containing subarrays with the indices of each of the folds.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def split_data_jet(x):
    """
    Splits the data depending on the value of the number of jets variable (feature #22)

    :param x: Input features.
    :return: Indices pertaining to jet numbers of 0, 1, or 2 and 3.
    """
    ind_0 = x[:, 22] == 0
    ind_1 = x[:, 22] == 1
    ind_2 = np.logical_or(x[:, 22] == 2, x[:, 22] == 3)

    return ind_0, ind_1, ind_2


def preprocess_data(x, nan_mode):
    """
    Perform all the pre-processing steps to the input data. Removes unnecessary features based on intercorrelations in
    the features and correlations with respect to the label data, as outlined in the report. Removes columns containing
    single unique elements and replaces nan entries by the selected mode.

    :param x: Input features.
    :param nan_mode: Appraoch for replacing the nan values. Can selected between 'mean', 'median' or 'mode'.
    :return: Preprocessed input features.
    """
    x = np.delete(x, [9, 15, 18, 20, 25, 28, 29], axis=1)

    x = remove_constant_columns(x)

    x = convert_nan(x, nan_mode)

    return x


def remove_outliers(x):
    """
    Filters outliers from the dataset by removing those within a certain number of standard deviations from the mean.

    :param x: Input features.
    :return: Filtered dataset with the outliers removed.
    """
    x_mean = np.mean(x, axis=0)
    x_sd = np.std(x, axis=0)

    lower_lim = x_mean - 3 * x_sd
    upper_lim = x_mean + 3 * x_sd

    testlower = np.any(x < lower_lim, axis=1)
    testupper = np.any(x > upper_lim, axis=1)

    outliers = np.logical_or(testlower, testupper)

    return x[~outliers], outliers


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


def transform_data(x_tr, x_te, degree):
    """
    Performs the data transformation and feature expansion on the input features. Concatenates the polynomial expansion
    basis, logarithmic basis (of positive columns), cross channel correlations, and the intercept term for the training
    and testing data.

    :param x_tr: Train input features.
    :param x_te: Test input features.
    :param degree: Degree of the polynomial basis.
    :return: Transformed, horizontally concatenated input feature matrix.
    """
    x_tr_cross = cross_channel_features(x_tr)
    x_te_cross = cross_channel_features(x_te)

    neg_cols_te = np.any(x_tr <= 0, axis=0)
    neg_cols_tr = np.any(x_te <= 0, axis=0)
    neg_cols = np.logical_or(neg_cols_te, neg_cols_tr)

    x_tr_log = np.log(x_tr[:, ~neg_cols])
    x_te_log = np.log(x_te[:, ~neg_cols])

    x_tr = build_poly(x_tr, degree)
    x_te = build_poly(x_te, degree)

    x_tr = np.concatenate((x_tr, x_tr_cross, x_tr_log), axis=1)
    x_te = np.concatenate((x_te, x_te_cross, x_te_log), axis=1)

    x_tr, tr_mean, tr_sd = standardize_data(x_tr)

    x_te[:, tr_sd > 0] = (x_te[:, tr_sd > 0] - tr_mean[tr_sd > 0]) / tr_sd[tr_sd > 0]
    x_te[:, tr_sd == 0] = x_te[:, tr_sd == 0] - tr_mean[tr_sd == 0]

    x_tr = np.concatenate((np.ones((x_tr.shape[0], 1)), x_tr), axis=1)
    x_te = np.concatenate((np.ones((x_te.shape[0], 1)), x_te), axis=1)

    return x_tr, x_te

# Implementation example
# x_tr = preprocess_data(x_tr, nan_mode=nan_mode)
# x_te = preprocess_data(x_te, nan_mode=nan_mode)
#
# x_tr, x_te = transform_data(x_tr, x_te, degree)
#
# loss_tr, w = method(y_tr, x_tr, **kwargs)
#
# loss_te = compute_mse(y_te, x_te, w)
#
# acc_tr = compute_accuracy(w, x_tr, y_tr, binary_mode=binary_mode)
# acc_te = compute_accuracy(w, x_te, y_te, binary_mode=binary_mode)
