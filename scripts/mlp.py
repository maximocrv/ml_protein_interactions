#!/bin/python
"""
This script implements the Multi-Layer Perceptron model.
"""

import itertools
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from constants import test_metrics
from scipy.stats import PearsonRConstantInputWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from train_helpers import train_adaptive, gen_loaders
from utilities import load_data, open_log, transform_data

# Whether to only train the model instead of performing cross-validation.
train_only = False


class MLP(torch.nn.Module):
    def __init__(self, input_dim, layers, nodes, dropout=0.0, do_batchnorm=False, output_dim=1):
        """
        Initializes MLP class.

        :param input_dim: Dimension of the inputs.
        :param layers: Number of hidden layers.
        :param nodes: Number of nodes per layer.
        :param dropout: Dropout conditions for each layer.
        :param do_batchnorm: Whether to perform batch normalization in each hidden layer.
        :param output_dim: Dimension of the output. Should always be 1.
        """
        super().__init__()
        self.input = nn.Linear(input_dim, nodes)
        self.hidden = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.do_batchnorm = do_batchnorm
        if do_batchnorm:
            self.batchnorm = nn.ModuleList()
        for _ in range(layers):
            self.hidden.append(nn.Linear(nodes, nodes))
            self.dropout.append(nn.Dropout(dropout))
            if do_batchnorm:
                self.batchnorm.append(nn.BatchNorm1d(nodes))
        self.out = nn.Linear(nodes, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Forward pass. Performs batchnorm if specified in the initialization.

        :param x: Input of the previously specified `input_dim`.
        """
        x = self.input(x)
        x = self.relu(x)

        for i, layer in enumerate(self.hidden):
            x = self.relu(layer(x))
            if self.do_batchnorm:
                x = self.dropout[i](self.batchnorm[i](x))
            else:
                x = self.dropout[i](x)

        return self.out(x)


if __name__ == "__main__":
    log = open_log('MLP')

    # print specific info about this run
    log.write('####train run 8 layers 128 nodes early stopping\n')

    if not torch.cuda.is_available():
        print('WARNING: using CPU.')
        log.write('\tWARNING: using CPU.\n')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_data()

    # clip data: u_lj mean and STD of mutant complexes
    X[:, 8] = np.clip(X[:, 8], 1e8, 5e9)  # u_lj_mut_mean
    X[:, 9] = np.clip(X[:, 9], 1e8, 5e10)  # u_lj_mut_std

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # standardize data
    x_train, x_test = transform_data(x_train, x_test, degree=1, log=False, cross=False)
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    print(f'######### x_test shape: {x_test.shape}')
    print(f'######### y_test shape: {y_test.shape}')
    log.write(f'\t######### x_test shape: {x_test.shape}\n')
    log.write(f'\t######### y_test shape: {y_test.shape}\n')
    log.write('\tx_train:\n')
    log.write(str(x_train))
    log.write('\n')

    criterion = torch.nn.MSELoss()

    # Baseline linear regression score
    reg = LinearRegression().fit(x_train, y_train)
    reg_y = reg.predict(x_test)

    linreg_score = test_metrics['pearsonr'](y_test, reg_y)
    print(f'Linear Regression test score: {linreg_score:10.5}')
    log.write(f'\tLinear Regression test score: {linreg_score:10.5}\n')

    # To supress scipy's pearsonr related warnings
    warnings.simplefilter("ignore", PearsonRConstantInputWarning)

    # Defining hyperparameters
    n_hidden_layers = [8]
    n_hidden_nodes = [128]
    learning_rates = [1e-3]
    dropouts = [0.25]
    batchnorm = [False]
    L2 = [0.0]
    batch_size = 256
    epochs = 200  # early stopping max consecutive epochs
    hyperparam_space = itertools.product(n_hidden_layers, n_hidden_nodes, learning_rates, dropouts, batchnorm, L2)
    kf = KFold(n_splits=5)

    opt_val_score = None
    best_heuristic = lambda x, y: y['MSE'] < x['MSE']

    if train_only:
        opt_params = (n_hidden_layers[0], n_hidden_nodes[0],
                      learning_rates[0], dropouts[0], batchnorm[0], L2[0])
        log.write('\thidden layers: {}, nodes per layer: {}, learning rate: {}, dropout: {}, do batchnorm: {}, L2: {}\n'
                  .format(opt_params[0], opt_params[1], opt_params[2], opt_params[3], opt_params[4], opt_params[5]))

    if not train_only:
        # begin cross-validation
        start_t = time.time()
        for params in hyperparam_space:
            # K-fold cross validation

            # output
            print('hidden layers: {}, nodes per layer: {}, learning rate: {}, dropout: {}, do batchnorm: {}, L2: {}'
                  .format(params[0], params[1], params[2], params[3], params[4], params[5]))
            log.write(
                '\thidden layers: {}, nodes per layer: {}, learning rate: {}, dropout: {}, do batchnorm: {}, L2: {}\n'
                .format(params[0], params[1], params[2], params[3], params[4], params[5]))

            train_losses_kf = []
            val_scores_kf = {key: [] for key in test_metrics}
            k = 0
            for train_index, val_index in kf.split(x_train):
                # output
                log.write(f'\tk_fold: {k}\n')
                print(f'k_fold: {k}')

                # None adds a new axis
                x_kftrain, y_kftrain = x_train[train_index], y_train[train_index, None]
                x_kfval, y_kfval = x_train[val_index], y_train[val_index, None]

                dataset_kftrain = gen_loaders(x_kftrain, y_kftrain, batch_size)
                dataset_kfval = gen_loaders(x_kfval, y_kfval, batch_size)

                model = MLP(
                    input_dim=x_train.shape[1],
                    layers=params[0],
                    nodes=params[1],
                    dropout=params[3],
                    do_batchnorm=params[4],
                    output_dim=1
                ).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=params[2], weight_decay=params[5])

                # train fold
                train_loss, val_scores = train_adaptive(model, criterion, dataset_kftrain, dataset_kfval, optimizer,
                                                        device, test_metrics, log, best_heuristic, epochs)

                train_losses_kf.append(train_loss)
                for key, fun in test_metrics.items():
                    val_scores_kf[key].append(val_scores[key])

                k += 1

            # average validation scores and training losses
            mean_train_loss = np.mean(train_losses_kf)
            mean_val_scores = {key: np.mean(val_scores_kf[key]) for key in test_metrics}

            # output
            val_scores_str = ' '.join([f'{k}={v:12.5g}' for k, v in mean_val_scores.items()])
            print('hyperparams.:{:16} train_loss={:12.5} val_scores: {}'.format(str(params), mean_train_loss,
                                                                                val_scores_str))
            log_str = "    {:16}    {:12}    {}\n".format('hyperparams', 'train_loss', '    '.join(test_metrics.keys()))
            log.write(log_str)
            log.write('-' * len(log_str) + '\n')
            log.write("    {:16}    {:12.5g}    {}\n\n".format(str(params), mean_train_loss,
                      '    '.join([f'{v:12.5g}' for v in mean_val_scores.values()])
            ))

            # find best hyperparameters according to the validation score
            if opt_val_score is None or mean_val_scores['pearsonr'] > opt_val_score:
                opt_val_score = mean_val_scores['pearsonr']
                opt_params = params

        # output run time
        end_t = time.time()
        log.write(f'\tfinished cross-validation in {end_t - start_t}s.\n')

        # output cross-validation results
        print(f'best hyperparams.:{str(opt_params)} with validation score={opt_val_score:12.5}')
        log.write(f'\tbest hyperparams.:{str(opt_params)} with validation score={opt_val_score:12.5}\n')

    # Final training
    start_t = time.time()

    model = MLP(input_dim=x_train.shape[1], layers=opt_params[0], nodes=opt_params[1], dropout=opt_params[3],
                do_batchnorm=opt_params[4], output_dim=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt_params[2], weight_decay=opt_params[5])

    # None adds a new axis
    dataset_train = gen_loaders(x_train, y_train[:, None], batch_size)
    dataset_test = gen_loaders(x_test, y_test[:, None], batch_size)

    train_epochs = 400  # early stopping max consecutive epochs

    # train final model
    train_loss, test_scores = train_adaptive(model, criterion, dataset_train, dataset_test, optimizer, device,
                                             test_metrics, log, best_heuristic, train_epochs)

    # output run time
    end_t = time.time()
    log.write(f'\tfinished final training in {end_t - start_t}s.\n')

    # output final training results
    test_scores_str = ' '.join([f'{k}={v:12.5g}' for k, v in test_scores.items()])
    print(f'final MSE train loss: {train_loss} final test scores:\n\t{test_scores_str}')
    log.write(f'final MSE train loss: {train_loss} final test score:\n\t{test_scores_str}\n')

    # just to remind us what the linear regression score was
    print(f'Linear Regression test score: {linreg_score:10.5}')

    log.close()
else:
    raise Exception('\tPlease execute this script directly.\n')
