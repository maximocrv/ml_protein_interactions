import math
import multiprocessing as mp
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from constants import wt_features_path, mut_features_path, skempi_csv, R
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# check if we are in a conda virtual env
try:
    os.environ["CONDA_DEFAULT_ENV"]
except KeyError:
    print("\tPlease init the conda environment!\n")
    exit(1)


def standardize(arr):
    return (arr - np.mean(arr)) / np.std(arr)


def siamese_preprocessing(pandas_row):
    name_wt = pandas_row[1].iloc[0]
    name_mut = pandas_row[1].iloc[0] + '_' + pandas_row[1].iloc[2].replace(',', '_')

    if not Path(mut_features_path + 'D_mat/' + name_mut + '.npy').exists():
        print(f'ERROR: {name_mut} does not exist.', '\n')
        return None
    if not Path(wt_features_path + 'D_mat/' + name_wt + '.npy').exists():
        print(f'ERROR: {name_wt} does not exist.', '\n')
        return None

    d_mat_wt = standardize(np.load(wt_features_path + 'D_mat/' + name_wt + '.npy'))
    u_lj_wt = standardize(np.load(wt_features_path + 'U_LJ/' + name_wt + '.npy'))
    u_el_wt = standardize(np.load(wt_features_path + 'U_el/' + name_wt + '.npy'))

    wt_arr = np.stack([d_mat_wt, u_lj_wt, u_el_wt])

    d_mat_mut = standardize(np.load(mut_features_path + 'D_mat/' + name_mut + '.npy'))
    u_lj_mut = standardize(np.load(mut_features_path + 'U_LJ/' + name_mut + '.npy'))
    u_el_mut = standardize(np.load(mut_features_path + 'U_el/' + name_mut + '.npy'))

    mut_arr = np.stack([d_mat_mut, u_lj_mut, u_el_mut])

    # calculate DDG
    A_wt = pandas_row[1]['Affinity_wt_parsed']
    A_mut = pandas_row[1]['Affinity_mut_parsed']

    # print(pandas_row[1]['Temperature'])
    temp = float(re.match("[0-9]*", pandas_row[1]['Temperature'])[0])
    if math.isnan(temp):
        raise ValueError('temperature should not be NaN.')

    DG_wt = R * temp * np.log(A_wt)
    DG_mut = R * temp * np.log(A_mut)
    DDG = DG_mut - DG_wt

    # debug print
    print(f'parsed {name_mut}')

    return np.stack([wt_arr, mut_arr]), DDG


def gen_loaders(x, y, batch_size):
    x_tensor, y_tensor = torch.from_numpy(x), torch.from_numpy(y)

    data = TensorDataset(x_tensor, y_tensor)

    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

    return loader


class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        # feature map output: [(W - K + 2P) / S] + 1
        # include batch norm, dropout (remember model.train() and model.eval() !!!)
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
                                  # output: (256-3)/2 + 1 =
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(8),

                                  nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(64),

                                  nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(512),

                                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2)
                                  )

        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
                                  # output: (256-3)/2 + 1 =
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(8),

                                  nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(64),

                                  nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(512),

                                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2)
                                  )

        # each output of self.cnn will have dimension 1024, so when concatenated we have 2048
        self.fc = nn.Sequential(  # nn.Linear(2048, 512),
            # nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Linear(2 * 512, 64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Linear(64, 1))

    def forward(self, x1):
        output1 = self.cnn1(x1[:, 0])
        output2 = output1.view(output1.size()[0], -1)

        output3 = self.cnn2(x1[:, 1])
        output4 = output3.view(output3.size()[0], -1)

        output5 = torch.cat((output2, output4), 1)

        return self.fc(output5)


def train(_model, _criterion, dataset_train, dataset_test, _optimizer, n_epochs):
    print("Starting training")
    for epoch in range(n_epochs):
        # Train an epoch
        _model.train()
        train_losses = []
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            logits = _model.forward(batch_x)
            loss = _criterion(logits, batch_y)
            train_losses.append(loss.item())

            # Compute the gradient
            _optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            _optimizer.step()

        train_loss = np.mean(train_losses)

        # Test the quality on the test set
        _model.eval()
        mse_test = []
        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            # TODO: solve allocation error in GPU
            prediction = _model(batch_x)
            mse_test.append(_criterion(prediction, batch_y).item())
            # ^^ could be source of memory leak
        print(
            "Epoch {} | Train loss: {:.5f} Test loss: {:.5f}".format(epoch, train_loss, sum(mse_test) / len(mse_test)))


if __name__ == '__main__':
    df = pd.read_csv(skempi_csv, sep=';')
    # df = df.iloc[:4096, :]

    # filter duplicated
    df = df[~df.duplicated(subset=["#Pdb", "Mutation(s)_cleaned"])]

    # remove without target
    df = df.dropna(subset=['Affinity_mut_parsed'])
    df = df.dropna(subset=['Affinity_wt_parsed'])
    df = df.dropna(subset=['Temperature'])

    input_list = []
    target_list = []

    n_non_existant = 0
    for data in mp.Pool(5).imap_unordered(siamese_preprocessing, df.iterrows()):
        if data is None:
            n_non_existant += 1
        else:
            input_list.append(data[0])
            target_list.append(data[1])

    print(f'{n_non_existant} PDBs do not have features.')

    input_arr = np.array(input_list).astype(np.float32)
    target_arr = np.array(target_list).astype(np.float32)[..., np.newaxis]

    x_tr, x_te, y_tr, y_te = train_test_split(input_arr, target_arr, test_size=0.2, random_state=42)
    # x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.25, random_state=42)

    train_data = gen_loaders(x_tr, y_tr, 16)
    # val_data = gen_loaders(x_val, y_val, 16)
    test_data = gen_loaders(x_te, y_te, 16)

    # model = HydraNet()
    num_epochs = 100
    learning_rate = 1e-3

    # If a GPU is available
    if not torch.cuda.is_available():
        print('WARNING: using CPU.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the logistic regression model with the Adam optimizer
    criterion = torch.nn.MSELoss()  # MSE loss for regression
    model_hydra = HydraNet().to(device)

    optimizer = torch.optim.Adam(model_hydra.parameters(), lr=learning_rate)
    # note that below validation data should actually be used...
    train(model_hydra, criterion, train_data, test_data, optimizer, num_epochs)

    model_hydra.eval().to("cpu")
    pred = model_hydra(torch.from_numpy(x_te))
    pred = pred.cpu().detach().numpy()
    R = pearsonr(y_te.squeeze(), pred.squeeze())[0]
    print(f'Pearson R score: {R:.5}', '\n')
    print(f'Test RMSE: {torch.sqrt(criterion(torch.tensor(pred), torch.tensor(y_te)))}')
