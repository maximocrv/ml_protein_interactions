"""
This script implements the HydraNet model.
"""
import math
import multiprocessing as mp
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from utilities import open_log
from train_helpers import train
from constants import wt_features_path, mut_features_path, skempi_csv, R, test_metrics


def hydra_loading(pandas_row):
    """
    Loads in a feature matrix with its corresponding target.

    :param pandas_row: Pandas row (used in cojunction with df.iterrows).
    :return: Tuple containing feature matrix and target DDG value.
    """
    name_wt = pandas_row[1].iloc[0]
    name_mut = pandas_row[1].iloc[0] + '_' + pandas_row[1].iloc[2].replace(',', '_')

    if not Path(mut_features_path + name_mut + '.npy').exists():
        print(f'ERROR: {name_mut} does not exist.', '\n')
        return None
    if not Path(wt_features_path + name_wt + '.npy').exists():
        print(f'ERROR: {name_wt} does not exist.', '\n')
        return None

    # Loading in feature matrices containing u_lj, u_el, d_mat (in this order!!)
    wt_arr = np.load(wt_features_path + name_wt + '.npy')
    mut_arr = np.load(mut_features_path + name_mut + '.npy')

    # Calculate DDG
    A_wt = pandas_row[1]['Affinity_wt_parsed']
    A_mut = pandas_row[1]['Affinity_mut_parsed']

    # print(pandas_row[1]['Temperature'])
    temp = float(re.match("[0-9]*", pandas_row[1]['Temperature'])[0])
    if math.isnan(temp):
        raise ValueError('temperature should not be NaN.')

    DG_wt = R * temp * np.log(A_wt)
    DG_mut = R * temp * np.log(A_mut)
    DDG = DG_mut - DG_wt

    # Debug print
    print(f'parsed {name_mut}')

    return np.stack([wt_arr, mut_arr]), DDG


class ProteinDataset(TensorDataset):
    def __init__(self, *tensors, augment_data, means, stds):
        """
        Initializes protein dataset class. Required due to random augmentations that are applied to the inputs, which
        are dual (due to input of both wild types and mutants).

        :param tensors: Torch tensors, namely feature matrix and target DDG. Tuple containing (x, y) data pairs. "x" has
        shape (N_SAMPLES, 2, C, H, W) - 2 arises from mut vs wt. "y" is one dimensional and contains N_SAMPLES entries.
        :param augment_data: Boolean to determine whether or not to augment the data (e.g. for training/ validation)
        :param means: Means of each matrix channel based on train set. Used to standardize the input.
        :param stds: Standard deviation of each matrix channel based on train set. Used to standardize the input.
        """
        super().__init__()
        self.tensors = tensors
        self.augment_data = augment_data
        self.means = means
        self.stds = stds

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        """
        Overriding __getitem__ method so it standardizes and returns an image with a random augmentation (i.e. rotation)

        :param index: Index of sample
        :return: Tuple with standardized and (potentially) augmented input features, and target DDG
        """
        wt_arr = self.tensors[0][index, 0]
        mut_arr = self.tensors[0][index, 1]

        wt_arr = (wt_arr - self.means[0]) / self.stds[0]
        mut_arr = (mut_arr - self.means[1]) / self.stds[1]

        n_rand = torch.rand(1)

        if self.augment_data:
           # if 0 <= n_rand < 0.25:
           #     wt_arr = torch.rot90(wt_arr, 1, (1, 2))
           #     mut_arr = torch.rot90(mut_arr, 1, (1, 2))

            if n_rand <= 0.5:
                wt_arr = torch.rot90(wt_arr, 2, (1, 2))
                mut_arr = torch.rot90(mut_arr, 2, (1, 2))

            #elif 0.5 <= n_rand < 0.75:
            #    wt_arr = torch.rot90(wt_arr, 3, (1, 2))
            #    mut_arr = torch.rot90(mut_arr, 3, (1, 2))

        self.tensors[0][index, 0] = wt_arr
        self.tensors[0][index, 1] = mut_arr

        return tuple(tensor[index] for tensor in self.tensors)


def gen_hydra_loaders(x, y, means, stds, batch_size, augment_data=False, mode=None):
    """
    Generates the data loader for training with the HydraNet.

    :param x: Input feature matrices
    :param y: Target data
    :param means: Means for the wild types and mutations for standardization
    :param stds: Standard deviations for the wild types and mutations for standardization
    :param batch_size: Batch size
    :param augment_data: Boolean to determine whether or not to augment the data
    :param mode: Either "train" or None, determines whether to shuffle the data or not
    :return:
    """
    x_tensor, y_tensor = torch.from_numpy(x), torch.from_numpy(y)
    means, stds = torch.from_numpy(means), torch.from_numpy(stds)

    data = ProteinDataset(x_tensor, y_tensor, augment_data=augment_data, means=means, stds=stds)

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)

    return loader


class HydraNet(nn.Module):
    def __init__(self):
        """
        Initializes HydraNet model, which is a double headed convolutional network which is then concatenated and fed
        through to a fully connected network. The architecture is described below.
        """
        super().__init__()
        # feature map output: [(W - K + 2P) / S] + 1
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
                                  # output: (256-3)/2 + 1 =
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(8),
                                  # nn.Dropout2d(p=0.2),

                                  nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(64),
                                  # nn.Dropout2d(p=0.2),
                                  # nn.AvgPool2d()

                                  # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                  # nn.ReLU(),
                                  # nn.MaxPool2d(2),
                                  # nn.BatchNorm2d(64),
                                  # nn.Dropout2d(p=0.15),
                                  # nn.AvgPool2d()

                                  nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(512),
                                  # nn.Dropout2d(p=0.2),

                                  # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
                                  # nn.ReLU(),
                                  # nn.MaxPool2d(2),
                                  # nn.BatchNorm2d(512),
                                  # nn.Dropout2d(p=0.15),

                                  nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  nn.BatchNorm2d(1024)
                                  )

        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
                                  # output: (256-3)/2 + 1 =
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(8),
                                  # nn.Dropout2d(p=0.2),

                                  nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(64),
                                  # nn.Dropout2d(p=0.2),
                                  # nn.AvgPool2d()

                                  # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                  # nn.ReLU(),
                                  # nn.MaxPool2d(2),
                                  # nn.BatchNorm2d(64),
                                  # nn.Dropout2d(p=0.15),
                                  # nn.AvgPool2d()

                                  nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(512),
                                  # nn.Dropout2d(p=0.2),

                                  # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
                                  # nn.ReLU(),
                                  # nn.MaxPool2d(2),
                                  # nn.BatchNorm2d(512),
                                  # nn.Dropout2d(p=0.15),

                                  nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  nn.BatchNorm2d(1024)
                                  )

        # each output of self.cnn will have dimension x, so when concatenated we have 2 * x
        self.fc = nn.Sequential(
            # nn.Linear(2048, 512),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(2 * 1024, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Dropout(p=0.1),
            nn.Linear(64, 1))

    def forward(self, x1):
        """
        Forward pass which completes the separate convolutions for the mutation and the wildtype, concatenates the
        outputs, and feeds them through a series of fully connected layers.

        :param x1: Input matrices of shape (N, 2, C, H, W)
        """
        output1 = self.cnn1(x1[:, 0])
        output1 = torch.mean(output1.view(output1.size(0), output1.size(1), -1), dim=2)
        output2 = output1.view(output1.size()[0], -1)

        output3 = self.cnn2(x1[:, 1])
        output3 = torch.mean(output3.view(output3.size(0), output3.size(1), -1), dim=2)
        output4 = output3.view(output3.size()[0], -1)

        output5 = torch.cat((output2, output4), 1)

        return self.fc(output5)


if __name__ == "__main__":
    # check if we are in a conda virtual env
    try:
        os.environ["CONDA_DEFAULT_ENV"]
    except KeyError:
        print("\tPlease init the conda environment!\n")
        exit(1)

    df = pd.read_csv(skempi_csv, sep=';')
    # df = df.iloc[:100, :]

    # filter duplicated
    df = df[~df.duplicated(subset=["#Pdb", "Mutation(s)_cleaned"])]

    # remove without target
    df = df.dropna(subset=['Affinity_mut_parsed'])
    df = df.dropna(subset=['Affinity_wt_parsed'])
    df = df.dropna(subset=['Temperature'])

    input_list = []
    target_list = []

    n_non_existant = 0
    for data in mp.Pool(5).imap_unordered(hydra_loading, df.iterrows()):
        if data is None:
            n_non_existant += 1
        else:
            input_list.append(data[0].astype(np.float32))
            target_list.append(data[1].astype(np.float32))

    print(f'{n_non_existant} PDBs do not have features.')

    input_arr = np.array(input_list)
    target_arr = np.array(target_list)[..., np.newaxis]

    x_tr, x_te, y_tr, y_te = train_test_split(input_arr, target_arr, test_size=0.2, random_state=42)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.25, random_state=42)

    means_wt = x_tr[:, 0].mean(axis=(0, 2, 3), keepdims=True)
    means_mut = x_tr[:, 1].mean(axis=(0, 2, 3), keepdims=True)
    means = np.vstack([means_wt, means_mut])

    stds_wt = x_tr[:, 0].std(axis=(2, 3), keepdims=True).mean(axis=0, keepdims=True)
    stds_mut = x_tr[:, 1].std(axis=(2, 3), keepdims=True).mean(axis=0, keepdims=True)
    stds = np.vstack([stds_wt, stds_mut])

    train_data = gen_hydra_loaders(x_tr, y_tr, means, stds, batch_size=16, augment_data=False, mode='train')
    val_data = gen_hydra_loaders(x_val, y_val, means, stds, batch_size=16, augment_data=False, mode=None)
    test_data = gen_hydra_loaders(x_te, y_te, means, stds, batch_size=16, augment_data=False, mode=None)

    # model = HydraNet()
    num_epochs = 100
    learning_rate = 1e-4

    # If a GPU is available
    if not torch.cuda.is_available():
        print('WARNING: using CPU.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log = open_log('hydranet')

    # Train the logistic regression model with the Adam optimizer
    criterion = torch.nn.MSELoss()  # MSE loss for regression
    model_hydra = HydraNet().to(device)
    optimizer = torch.optim.Adam(model_hydra.parameters(), lr=learning_rate)
    train_loss, eval_scores = train(model_hydra, criterion, train_data, val_data, optimizer, num_epochs, device,
                                    test_metrics, log)

#     model_hydra.eval().to("cpu")
    model_hydra.eval()

    log.write(str(model_hydra.parameters))
    log.write(str(optimizer))

    n = 0
    test_rmse = []
    test_pearson = []
    pred_te = []
    for batch_x, batch_y in test_data:
        batch_x = batch_x.to(device)
        preds = model_hydra(batch_x)
        
        preds = preds.cpu().detach().numpy().ravel()
        batch_y = batch_y.detach().numpy().ravel()
        
        pred_te.append(preds)

        batch_rmse = np.sqrt(np.mean(np.square(preds - batch_y)))
        test_rmse.append(batch_rmse)

        batch_pearson = pearsonr(preds, batch_y)[0]
        test_pearson.append(batch_pearson)

    R = pearsonr(y_te.squeeze(), np.concatenate(pred_te).squeeze())[0]
    print(f'Pearson R test score: {R:.5}', '\n')
    print(f'Test RMSE: {np.mean(test_rmse)}')

    log.write(str(pred_te))
    log.write(f"Test Pearson R: {R}")
    log.write(f"Test RMSE: {np.mean(test_rmse)}")

    log.close()
