import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_opt import BayesianOptimization
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

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
        

        return torch.sigmoid(self.out(x))


def blob_label(y, label, loc): # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target


def hyperparameter_tuning(hyperparameter_dict):
    pass


X, y = make_moons(noise=0.3, random_state=0) 
X = StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

h=0.02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.axes()
ax.set_title("Input data")
# Plot the training points
ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
           edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
#plt.show()

x_train =torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

print(f'######### x_test shape: {x_test.shape}')

model = MLP(input_dim=x_train.shape[1], layers=6, nodes=50, output_dim=1)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.eval()
y_pred = model(x_test)
before_train = criterion(y_pred.squeeze(), y_test)
print(model)

model.train()
epochs=20

for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(x_train)
    
    loss = criterion(y_pred.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    print(f'model loss at epoch {epoch} : {loss.item()}') 

