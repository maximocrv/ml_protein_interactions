#!/bin/python

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from bayes_opt import BayesianOptimization
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
# from matplotlib.colors import ListedColormap
import pandas as pd

import itertools

#https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
#https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
class MLP(torch.nn.Module):
    def __init__(self, input_dim, layers, nodes, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, nodes)
        self.hidden = nn.ModuleList()
        for l in range(layers):
            self.hidden.append(nn.Linear(nodes, nodes))
        self.out = nn.Linear(nodes, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        
        for l in self.hidden:
            x = self.relu(l(x))
        
        return self.out(x)


# def blob_label(y, label, loc): # assign labels
#     target = np.copy(y)
#     for l in loc:
#         target[y == l] = label
#     return target


# def hyperparameter_tuning(hyperparameter_dict):
#     pass

# def gen_model_data(x: np.array, y: np.array, random_state=1):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=random_state)
# 
#     return x_train, y_train, x_val, y_val, x_test, y_test

# X, y = make_moons(noise=0.3, random_state=0) 
# X = StandardScaler().fit_transform(X)

# h=0.02
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))

# # just plot the dataset first
# cm = plt.cm.RdBu
# cm_bright = ListedColormap(['#FF0000', '#0000FF'])
# ax = plt.axes()
# ax.set_title("Input data")
# # Plot the training points
# ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright,
#            edgecolors='k')
# # Plot the testing points
# ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
#            edgecolors='k')
# ax.set_xlim(xx.min(), xx.max())
# ax.set_ylim(yy.min(), yy.max())
# ax.set_xticks(())
# ax.set_yticks(())
# #plt.show()

# model.eval()
# y_pred = model(x_test)
# before_train = criterion(y_pred.squeeze(), y_test)
# print(model)


# If a GPU is available (should be on Colab, we will use it)
if not torch.cuda.is_available():
    print('WARNING: using CPU.')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    df = pd.read_csv('../data/mlp_features.csv')
    df.drop(columns='mut', inplace=True)
    df = (df - df.mean(axis=0)) / df.std(axis=0)
    #print(df.head())
    return np.array(df.iloc[:, 1:-1], dtype=np.float32), np.array(df.iloc[:, -1], dtype=np.float32)
    #return df[:, 1:-1], df[:, -1]

def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size, optimizer, val_metric):
    train_data = TensorDataset(x_train.to(device), y_train.to(device))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    total_train_loss = []
    total_val_loss = []
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    for epoch in range(epochs):
        # train
        model.train()
        for batch_x, batch_y in train_loader:
            # forward pass
            y_pred = model.forward(batch_x)
            #print(y_pred)
            # gradient
            optimizer.zero_grad()
            loss = criterion(y_pred.squeeze(), batch_y)
            loss.backward()

            optimizer.step()

            total_train_loss.append(loss.item())

        # validation
        model.eval()
        # forward pass
        prediction = model.forward(x_val)
        pred_np  = prediction.cpu().detach().numpy().squeeze()
        y_val_np = y_val.cpu().detach().numpy()
        # debug print
        #print(f'std: {np.std(pred_np):10.5} {np.std(y_val_np):10.5}    shapes: {pred_np.shape} {y_val_np.shape}')
        total_val_loss.append( val_metric(y_val_np, pred_np) )
    
        if epoch%50 == 0:
            print(f'epoch {epoch:5} : train MSE loss={loss.item():10.5} val score={total_val_loss[-1]:10.5}') 

    return np.mean(total_train_loss), np.mean(total_val_loss)

X, y = load_data()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


x_train = torch.from_numpy(x_train)
x_test  = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test  = torch.from_numpy(y_test)
print(f'######### x_test shape: {x_test.shape}')
print(f'######### y_test shape: {y_test.shape}')

criterion = torch.nn.MSELoss()
# (real_y, pred_y)
test_metric = lambda x, y: pearsonr(x, y)[0] # r2_score 

reg = LinearRegression().fit(x_train, y_train)
reg_y = reg.predict(x_test)

linreg_score = test_metric(y_test, reg_y)
print(f'Linear Regression test score: {linreg_score:10.5}')

n_hidden_layers = [8, 16, 32]
n_hidden_nodes  = [16, 32, 64]
learning_rates = [1e-3, 1e-2, 1e-1]
batch_size = 32
hyperparam_space = itertools.product(n_hidden_layers, n_hidden_nodes, learning_rates)
kf = KFold(n_splits=5)
epochs=400
CV_scores=[]
opt_val_score=None
for params in hyperparam_space:
    # K-fold cross validation
    print(f'hidden layers: {params[0]}, nodes per layer: {params[1]}, learning rate: {params[2]}')
    train_losses_kf = []
    val_losses_kf = []
    for train_index, val_index in kf.split(x_train):
        x_kftrain, y_kftrain = x_train[train_index], y_train[train_index]
        x_kfval,  y_kfval    = x_train[val_index],   y_train[val_index]
        
        model = MLP(input_dim = x_train.shape[1], layers=params[0], nodes=params[1], output_dim=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params[2])

        train_loss, val_loss = train_model(model, x_kftrain, y_kftrain, x_kfval, y_kfval, \
                epochs, batch_size, optimizer, test_metric)

        train_losses_kf.append(train_loss)
        val_losses_kf.append(val_loss)

    mean_train_loss = np.mean(train_losses_kf)
    mean_val_score = np.mean(val_losses_kf)

    print(f'hyperparams.:{str(params):16} train_loss={mean_train_loss:12.5} val_score={mean_val_score:12.5}')

    CV_scores.append(mean_val_score)

    # best hyperparameters
    if opt_val_score is None or mean_val_score > opt_val_score:
        opt_val_score = mean_val_score
        opt_params = params


print(f'best hyperparams.:{str(opt_params)} with validation score={opt_val_score:12.5}')

model = MLP(input_dim = x_train.shape[1], layers=opt_params[0], nodes=opt_params[1], output_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt_params[2])

train_loss, test_loss = train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size, optimizer, test_metric)

print(f'final MSE train loss: {train_loss} final test score: {test_loss}')

print(f'Linear Regression test score: {linreg_score:10.5}')
