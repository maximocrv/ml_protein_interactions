import os
import re
import math
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr

# check if we are in a conda virtual env
try:
   os.environ["CONDA_DEFAULT_ENV"]
except KeyError:
   print("\tPlease init the conda environment!\n")
   exit(1)

def standardize(arr):
    return (arr - np.mean(arr)) / np.std(arr)

DATA_PATH='../data/'
SKEMPI_CSV=DATA_PATH + 'skempi_v2_cleaned.csv'
WT_FEATURE_PATH=DATA_PATH + 'openmm/'
MUT_FEATURE_PATH=DATA_PATH + 'openmm_mutated/'

R = (8.314/4184)  # kcal mol^-1 K^-1

def siamese_preprocessing(pandas_row):
    name_wt  = pandas_row[1].iloc[0]
    name_mut = pandas_row[1].iloc[0] + '_' + pandas_row[1].iloc[2].replace(',', '_')

    if not Path(MUT_FEATURE_PATH + 'D_mat/' + name_mut + '.npy').exists():
        print(f'ERROR: {name_mut} does not exist.', '\n')
        return None
    if not Path(WT_FEATURE_PATH + 'D_mat/' + name_wt + '.npy').exists():
        print(f'ERROR: {name_wt} does not exist.', '\n')
        return None

    d_mat_wt  = standardize(np.load(WT_FEATURE_PATH + 'D_mat/' + name_wt +'.npy'))
    u_lj_wt   = standardize(np.load(WT_FEATURE_PATH + 'U_LJ/' + name_wt +'.npy'))
    u_el_wt   = standardize(np.load(WT_FEATURE_PATH + 'U_el/' + name_wt +'.npy'))

    wt_arr = np.stack([d_mat_wt, u_lj_wt, u_el_wt])

    d_mat_mut = standardize(np.load(MUT_FEATURE_PATH + 'D_mat/' + name_mut + '.npy'))
    u_lj_mut  = standardize(np.load(MUT_FEATURE_PATH + 'U_LJ/' + name_mut + '.npy'))
    u_el_mut  = standardize(np.load(MUT_FEATURE_PATH + 'U_el/' + name_mut + '.npy'))

    mut_arr = np.stack([d_mat_mut, u_lj_mut, u_el_mut])

    # calculate DDG
    A_wt  = pandas_row[1]['Affinity_wt_parsed']
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

    return (np.stack([wt_arr, mut_arr]), DDG)

def gen_loaders(x_tr, x_te, y_tr, y_te, batch_size):    
    x_train_tensor, x_test_tensor = torch.from_numpy(x_tr), torch.from_numpy(x_te)
    y_train_tensor, y_test_tensor = torch.from_numpy(y_tr), torch.from_numpy(y_te)
    
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    
    test_data = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

#if __name__ == '__main__':
df = pd.read_csv(SKEMPI_CSV, sep=';')

df=df.iloc[:4096,:]

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
target_arr = np.array(target_list).astype(np.float32)[...,np.newaxis]
x_tr, x_te, y_tr, y_te = train_test_split(input_arr, target_arr, test_size=0.2, random_state=1)
train_data, test_data = gen_loaders(x_tr, x_te, y_tr, y_te, 32)


class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        # feature map output: [(W - K + 2P) / S] + 1
        # include batch norm, dropout (remember model.train() and model.eval() !!!)
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2), #output: (256-3)/2 + 1 =
                                 nn.ReLU(),
                                 nn.MaxPool2d(3, stride=2),

                                 nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=2),
                                 nn.ReLU(),
                                 nn.MaxPool2d(3, stride=2),

                                 nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d(3, stride=2),
                                 nn.Dropout2d(p=0.5),

                                 nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2)
                                )

        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2), #output: (256-3)/2 + 1 =
                                 nn.ReLU(),
                                 nn.MaxPool2d(3, stride=2),

                                 nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=2),
                                 nn.ReLU(),
                                 nn.MaxPool2d(3, stride=2),

                                 nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d(3, stride=2),
                                 nn.Dropout2d(p=0.5),

                                 nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2)
                                )

        # each output of self.cnn will have dimension 1024, so when concatenated we have 2048
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Linear(32, 1))

    def forward(self, x1):
        output1 = self.cnn1(x1[:, 0])
        output2 = output1.view(output1.size()[0], -1)

        output3 = self.cnn2(x1[:, 1])
        output4 = output3.view(output3.size()[0], -1)

        output5 = torch.cat((output2, output4), 1)

        return self.fc(output5)

def train(model, criterion, dataset_train, dataset_test, optimizer, num_epochs):
    """
    @param model: torch.nn.Module
    @param criterion: torch.nn.modules.loss._Loss
    @param dataset_train: torch.utils.data.DataLoader
    @param dataset_test: torch.utils.data.DataLoader
    @param optimizer: torch.optim.Optimizer
    @param num_epochs: int
    """
    print("Starting training")
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        train_losses=[]
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            logits = model.forward(batch_x)
            loss = criterion(logits, batch_y)
            train_losses.append(loss.item())

            # Compute the gradient
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()

        train_loss=np.mean(train_losses)

        # Test the quality on the test set
        model.eval()
        mse_test = []
        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            # TODO: solve allocation error in GPU
            prediction = model(batch_x)
            mse_test.append(criterion(prediction, batch_y))

        print("Epoch {} | Train loss: {:.5f} Test loss: {:.5f}".format(epoch, train_loss, sum(mse_test).item()/len(mse_test)))

model = HydraNet()
num_epochs = 200
learning_rate = 1e-4

# If a GPU is available 
if not torch.cuda.is_available():
    print('WARNING: using CPU.')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the logistic regression model with the Adam optimizer
criterion = torch.nn.MSELoss() # MSE loss for regression
model_hydra = HydraNet().to(device)

optimizer = torch.optim.Adam(model_hydra.parameters(), lr=learning_rate)
train(model_hydra, criterion, train_data, test_data, optimizer, num_epochs)

model_hydra.eval()
pred = model_hydra(torch.from_numpy(x_te).to(device))
pred = pred.cpu().detach().numpy()
R = pearsonr(y_te.squeeze(), pred.squeeze())[0]
print(f'Pearson R score: {R:.5}')
