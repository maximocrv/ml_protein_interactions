"""
This script implements the HydraNet model.
"""
import math
import multiprocessing as mp
import os
import re
from pathlib import Path

from tqdm import tqdm
from skimage.transform import rotate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from utilities import open_log
from train_helpers import train
from constants import wt_features_path, mut_features_path, skempi_csv, R, test_metrics


# check if we are in a conda virtual env
try:
    os.environ["CONDA_DEFAULT_ENV"]
except KeyError:
    print("\tPlease init the conda environment!\n")
    exit(1)


def hydra_loading(pandas_row):
    name_wt = pandas_row[1].iloc[0]
    name_mut = pandas_row[1].iloc[0] + '_' + pandas_row[1].iloc[2].replace(',', '_')

    if not Path(mut_features_path +  name_mut + '.npy').exists():
        print(f'ERROR: {name_mut} does not exist.', '\n')
        return None
    if not Path(wt_features_path + name_wt + '.npy').exists():
        print(f'ERROR: {name_wt} does not exist.', '\n')
        return None
    
    # u_lj, u_el, d_mat (in this order!!)
    wt_arr = np.load(wt_features_path + name_wt + '.npy')
#     wt_arr[0] = log_standardize(np.clip(wt_arr[0], None, 1e12))
#     wt_arr[1] = standardize(wt_arr[1])
#     wt_arr[2] = standardize(wt_arr[2])
    
    
    mut_arr = np.load(mut_features_path + name_mut + '.npy')
#     mut_arr[0] = log_standardize(np.clip(mut_arr[0], None, 1e12))
#     mut_arr[1] = standardize(mut_arr[1])
#     mut_arr[2] = standardize(mut_arr[2])
    
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


class ProteinDataset(TensorDataset):
    # self.tensors is a tuple containing (x, y) data pairs. x has shape (N, 2, C, H, W) - 2 arises from mut vs wt
    # y is one dimensional and contains N entries
    def __init__(self, *tensors, augment_data, means, stds):
        super().__init__()
        self.tensors = tensors
        self.augment_data = augment_data
        self.means = means
        self.stds = stds
            
            
    def __len__(self):
        return self.tensors[0].size(0)
    
    def __getitem__(self, index):
        wt_arr = self.tensors[0][index, 0]
        mut_arr = self.tensors[0][index, 1]
        
#         print(self.tensors[0][index, 0].shape, self.tensors[1][index, 0].shape)
        
        wt_arr = (wt_arr - self.means[0]) / self.stds[0] 
        mut_arr = (mut_arr - self.means[1]) / self.stds[1]
        
        n_rand = torch.rand(1)
        
        if self.augment_data:
            if 0 <= n_rand < 0.25:
                wt_arr = torch.rot90(wt_arr, 1, (1, 2))
                mut_arr = torch.rot90(mut_arr, 1, (1, 2))
            
            elif 0.25 <= n_rand < 0.25:
                wt_arr = torch.rot90(wt_arr, 2, (1, 2))
                mut_arr = torch.rot90(mut_arr, 2, (1, 2))
                    
            elif 0.5 <= n_rand < 0.75:
                wt_arr = torch.rot90(wt_arr, 3, (1, 2))
                mut_arr = torch.rot90(mut_arr, 3, (1, 2))
                
#             else:
#                 wt_arr = wt_arr
#                 mut_arr = mut_arr
        
#         print(wt_arr.mean(), mut_arr.mean())
#         print(self.tensors[0].shape, self.tensors[1].shape)

        self.tensors[0][index, 0] = wt_arr
        self.tensors[0][index, 1] = mut_arr

        return tuple(tensor[index] for tensor in self.tensors)


x_tr, x_te, y_tr, y_te = train_test_split(input_arr, target_arr, test_size=0.2, random_state=42)
x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.25, random_state=42)

means_wt = x_tr[:, 0].mean(axis=(0, 2, 3), keepdims=True)
means_mut = x_tr[:, 1].mean(axis=(0, 2, 3), keepdims=True)
means = np.vstack([means_wt, means_mut])

stds_wt = x_tr[:, 0].std(axis=(2, 3), keepdims=True).mean(axis=0, keepdims=True)
stds_mut = x_tr[:, 1].std(axis=(2, 3), keepdims=True).mean(axis=0, keepdims=True)
stds = np.vstack([stds_wt, stds_mut])


def gen_loaders(x, y, means, stds, batch_size, augment_data=False, mode=None):
    x_tensor, y_tensor = torch.from_numpy(x), torch.from_numpy(y)
    means, stds = torch.from_numpy(means), torch.from_numpy(stds)

    data = ProteinDataset(x_tensor, y_tensor, augment_data=augment_data, means=means, stds=stds)

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
            
    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)

    return loader

    
train_data = gen_loaders(x_tr, y_tr, means, stds, batch_size=16, augment_data=False, mode='train')
val_data = gen_loaders(x_val, y_val, means, stds, batch_size=16, augment_data=False, mode=None)
test_data = gen_loaders(x_te, y_te, means, stds, batch_size=16, augment_data=False, mode=None)

    
class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        # feature map output: [(W - K + 2P) / S] + 1
        # include batch norm, dropout (remember model.train() and model.eval() !!!)
#         self.fc_in = 1
        self.cnn1 =  nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
                                  # output: (256-3)/2 + 1 =
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(8),
#                                   nn.Dropout2d(p=0.2),

                                  nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(64),
#                                   nn.Dropout2d(p=0.2),
                                  #nn.AvgPool2d()

                                  #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                  #nn.ReLU(),
                                  #nn.MaxPool2d(2),
                                  #nn.BatchNorm2d(64),
#                                   nn.Dropout2d(p=0.15),
                                  #nn.AvgPool2d()

                                  nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(256),
#                                   nn.Dropout2d(p=0.2),

                                  #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
                                  #nn.ReLU(),
                                  #nn.MaxPool2d(2),
                                  #nn.BatchNorm2d(512),
#                                   nn.Dropout2d(p=0.15),

                                  nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  nn.BatchNorm2d(512)
                                  )




        
        self.cnn2 =  nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
                                  # output: (256-3)/2 + 1 =
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(8),
#                                   nn.Dropout2d(p=0.2),

                                  nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(64),
#                                   nn.Dropout2d(p=0.2),
                                  #nn.AvgPool2d()

                                  #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                  #nn.ReLU(),
                                  #nn.MaxPool2d(2),
                                  #nn.BatchNorm2d(64),
#                                   nn.Dropout2d(p=0.15),
                                  #nn.AvgPool2d()

                                  nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, stride=2),
                                  nn.BatchNorm2d(256),
#                                   nn.Dropout2d(p=0.2),

                                  #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
                                  #nn.ReLU(),
                                  #nn.MaxPool2d(2),
                                  #nn.BatchNorm2d(512),
#                                   nn.Dropout2d(p=0.15),

                                  nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  nn.BatchNorm2d(512)
                                  )


 
        
        # each output of self.cnn will have dimension 1024, so when concatenated we have 2048
        self.fc = nn.Sequential( 
            # nn.Linear(2048, 512),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(2 * 512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
#             nn.Dropout(p=0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
#             nn.Dropout(p=0.1),
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
        output1 = self.cnn1(x1[:, 0])
        output1 = torch.mean(output1.view(output1.size(0), output1.size(1), -1), dim=2)
        output2 = output1.view(output1.size()[0], -1)

#         # next should be equivalent to output2
#         trial = torch.mean(output1.view(output1.size(0), output1.size(1), -1), dim=2)
#         trial = trial.view(trial.size()[0], -1)

        output3 = self.cnn2(x1[:, 1])
        output3 = torch.mean(output3.view(output3.size(0), output3.size(1), -1), dim=2)
        output4 = output3.view(output3.size()[0], -1)

        output5 = torch.cat((output2, output4), 1)
#         print(output5.shape[1])

#         return output1.shape, output2.shape, trial.shape
        return self.fc(output5)


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
# criterion = torch.nn.SmoothL1Loss()
model_hydra = HydraNet().to(device)

# test out using new loss - cheeks tbh
# test out using repeated output feature maps + higher learning rate!! #########################
# test out with higher weight decay
# compare using one vs two heads - TBD


# perhaps add larger degree of weight decay
optimizer = torch.optim.Adam(model_hydra.parameters(), lr=learning_rate)
# note that below validation data should actually be used...
train_loss, eval_scores = train(model_hydra, criterion, train_data, val_data, optimizer, num_epochs, device, test_metrics, log)

model_hydra.eval().to("cpu")

log.write(str(model_hydra.parameters))
log.write(str(optimizer))

n = 0
test_rmse = []
test_pearson = []
pred_te = []
for batch_x, batch_y in test_data:
    preds = model_hydra(batch_x).detach()
    pred_te.append(preds.numpy().squeeze())
    
    batch_mse = torch.sqrt(torch.mean(torch.square(preds - batch_y))).item()
    test_rmse.append(batch_mse)
    
    batch_pearson = pearsonr(preds.detach().numpy().squeeze(), batch_y.detach().numpy().squeeze())[0]
    test_pearson.append(batch_pearson)
    
R = pearsonr(y_te.squeeze(), np.concatenate(pred_te).squeeze())[0]
print(f'Pearson R test score: {R:.5}', '\n')
print(f'Test RMSE: {np.mean(test_rmse)}')

log.write(str(pred_te), '\n')
log.write(f"Test Pearson R: {R}")
log.write(f"Test RMSE: {np.mean(test_rmse)}")