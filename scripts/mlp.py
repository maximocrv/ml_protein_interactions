#!/bin/python

import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold

from utilities import load_data


def gen_model_data(x: np.array, y: np.array, random_state=1):
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, random_state=random_state)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_tr, y_tr, test_size=0.25, random_state=random_state)

    return x_tr, y_tr, x_val, y_val, x_te, y_te


class MLP(torch.nn.Module):
    def __init__(self, input_dim, layers, nodes, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, nodes)
        self.hidden = nn.ModuleList()
        for _ in range(layers):
            self.hidden.append(nn.Linear(nodes, nodes))
        self.out = nn.Linear(nodes, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)

        for layer in self.hidden:
            x = self.relu(layer(x))

        return self.out(x)


#def train(_model, _criterion, dataset_train, dataset_test, _optimizer, n_epochs):
def train(_model, x_tr, y_tr, x_val, y_val, n_epochs,
          _batch_size, _optimizer, _criterion, val_metric):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = TensorDataset(x_tr.to(device), y_tr.to(device))
    train_loader = DataLoader(dataset=train_data, batch_size=_batch_size)

    total_train_loss = []
    total_val_loss = []

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    for epoch in range(n_epochs):
        _model.train()
        for batch_x, batch_y in train_loader:

            y_pred = _model.forward(batch_x)

            _optimizer.zero_grad()
            loss = _criterion(y_pred.squeeze(), batch_y)
            loss.backward()

            _optimizer.step()

            total_train_loss.append(loss.item())

        _model.eval()
        prediction = _model.forward(x_val)
        pred_np = prediction.cpu().detach().numpy().squeeze()
        y_val_np = y_val.cpu().detach().numpy()

        # debug print
        # print(f'std: {np.std(pred_np):10.5} {np.std(y_val_np):10.5}    shapes: {pred_np.shape} {y_val_np.shape}')

        total_val_loss.append(val_metric(y_val_np, pred_np))

        if epoch % 50 == 0:
            print(
                f'epoch {epoch:5} : train MSE loss={loss.item():10.5} val score={total_val_loss[-1]:10.5}')

    return np.mean(total_train_loss), np.mean(total_val_loss)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print('WARNING: using CPU.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42)

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    print(f'######### x_test shape: {x_test.shape}')
    print(f'######### y_test shape: {y_test.shape}')

    criterion = torch.nn.MSELoss()
    # (real_y, pred_y)
    def test_metric(x, y): return pearsonr(x, y)[0]  # r2_score

    reg = LinearRegression().fit(x_train, y_train)
    reg_y = reg.predict(x_test)

    linreg_score = test_metric(y_test, reg_y)
    print(f'Linear Regression test score: {linreg_score:10.5}')

    n_hidden_layers = [8, 16, 32]
    n_hidden_nodes = [16, 32, 64]
    learning_rates = [1e-3, 1e-2, 1e-1]
    batch_size = 32
    hyperparam_space = itertools.product(
        n_hidden_layers, n_hidden_nodes, learning_rates)
    kf = KFold(n_splits=5)
    epochs = 400
    CV_scores = []
    opt_val_score = None
    for params in hyperparam_space:
        # K-fold cross validation
        print(
            f'hidden layers: {params[0]}, nodes per layer: {params[1]}, learning rate: {params[2]}')
        train_losses_kf = []
        val_losses_kf = []
        for train_index, val_index in kf.split(x_train):
            x_kftrain, y_kftrain = x_train[train_index], y_train[train_index]
            x_kfval, y_kfval = x_train[val_index],   y_train[val_index]

            model = MLP(
                input_dim=x_train.shape[1],
                layers=params[0],
                nodes=params[1],
                output_dim=1
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params[2])

            train_loss, val_loss = \
                train(model, x_kftrain, y_kftrain, x_kfval, y_kfval,
                      epochs, batch_size, optimizer, criterion, test_metric)

            train_losses_kf.append(train_loss)
            val_losses_kf.append(val_loss)

        mean_train_loss = np.mean(train_losses_kf)
        mean_val_score = np.mean(val_losses_kf)

        print(
            f'hyperparams.:{str(params):16} train_loss={mean_train_loss:12.5} val_score={mean_val_score:12.5}')

        CV_scores.append(mean_val_score)

        # best hyperparameters
        if opt_val_score is None or mean_val_score > opt_val_score:
            opt_val_score = mean_val_score
            opt_params = params

    print(
        f'best hyperparams.:{str(opt_params)} with validation score={opt_val_score:12.5}')

    model = MLP(input_dim=x_train.shape[1], layers=opt_params[0],
                nodes=opt_params[1], output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_params[2])

    train_loss, test_loss = \
        train(model, x_train, y_train, x_test, y_test,
              epochs, batch_size, optimizer, criterion, test_metric)

    print(f'final MSE train loss: {train_loss} final test score: {test_loss}')

    print(f'Linear Regression test score: {linreg_score:10.5}')
