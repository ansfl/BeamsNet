################## Imports ##################

import torch

import torch.nn as nn
import numpy as np
import pandas as pd
from numpy import load
from numpy import transpose
from numpy import cos
from numpy import sin
from numpy.random.mtrand import random
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
from torch.utils.data import Dataset, DataLoader
import os

# Seeds
torch.manual_seed(0)
np.random.seed(0)


################## DNN ##################
class BeamsNetV2(nn.Module):
    def __init__(self):
        super(BeamsNetV2, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=T, out_channels=6,
                      kernel_size=2, stride=1),
            nn.Tanh(),
        )
        self.FC_ConvToFc = nn.Sequential(
            nn.Linear(18, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.ReLU(),
        )
        self.FC_output = nn.Sequential(
            nn.Linear(6, 3),

        )
        self.initialize_weights()

    def forward(self, x, y):
        y = self.conv_layer(y)
        y = torch.flatten(y, 1)
        y = self.FC_ConvToFc(y)
        x = torch.column_stack((x, y))
        x = self.FC_output(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

################## Functions ##################

def RMSE(true, predicted, LS):
    true = LA.norm(true, axis=1)
    predicted = LA.norm(predicted, axis=1)
    LS = LA.norm(LS, axis=1)
    rmse_ls = np.sqrt(mean_squared_error(true, LS))
    rmse_predicted = np.sqrt(mean_squared_error(true, predicted))
    improv = 100 * (1 - (rmse_predicted / rmse_ls))
    return rmse_ls, rmse_predicted, improv


def MAE(true, predicted, LS):
    true = LA.norm(true, axis=1)
    predicted = LA.norm(predicted, axis=1)
    LS = LA.norm(LS, axis=1)
    mae_ls = np.sum(np.abs(LS - true)) / len(true)
    mse_predicted = np.sum(np.abs(predicted - true)) / len(true)
    return mae_ls, mse_predicted


def NSE_R2(true, predicted, LS):
    true = LA.norm(true, axis=1)
    predicted = LA.norm(predicted, axis=1)
    LS = LA.norm(LS, axis=1)
    true_avg = np.mean(true)
    temp_ls = np.sum((LS - true) ** 2) / np.sum((true - true_avg) ** 2)
    r2_ls = 1 - temp_ls
    temp_ls = np.sum((predicted - true) ** 2) / np.sum((true - true_avg) ** 2)
    r2_predicted = 1 - temp_ls
    return r2_ls, r2_predicted


def VAF(true, predicted, LS):
    true = LA.norm(true, axis=1)
    predicted = LA.norm(predicted, axis=1)
    LS = LA.norm(LS, axis=1)
    true_var = np.var(true)
    temp_ls = np.var(true - LS)
    r2_ls = (1 - temp_ls / true_var) * 100
    temp_predicted = np.var(true - predicted)
    r2_predicted = (1 - temp_predicted / true_var) * 100
    return r2_ls, r2_predicted


################## Main ##################
# load
path = os.getcwd()
path = os.path.abspath(os.path.join(path, os.pardir))
V = load(path + '\dataset\Test\V_test.npy')

# DVL speed to beams
b1 = np.array([cos((45 + 0 * 90) * np.pi / 180) * sin(20 * np.pi / 180),
               sin((45 + 0 * 90) * np.pi / 180) * sin(20 * np.pi / 180), cos(20 * np.pi / 180)])
b2 = np.array([cos((45 + 1 * 90) * np.pi / 180) * sin(20 * np.pi / 180),
               sin((45 + 1 * 90) * np.pi / 180) * sin(20 * np.pi / 180), cos(20 * np.pi / 180)])
b3 = np.array([cos((45 + 2 * 90) * np.pi / 180) * sin(20 * np.pi / 180),
               sin((45 + 2 * 90) * np.pi / 180) * sin(20 * np.pi / 180), cos(20 * np.pi / 180)])
b4 = np.array([cos((45 + 3 * 90) * np.pi / 180) * sin(20 * np.pi / 180),
               sin((45 + 3 * 90) * np.pi / 180) * sin(20 * np.pi / 180), cos(20 * np.pi / 180)])
A = np.array([b1, b2, b3, b4]).reshape((4, 3))
p_inv = np.matmul(inv(np.matmul(transpose(A), A)), transpose(A))
beams = np.zeros((len(V[0, :]), 4))
for i in range(0, len(V[0, :])):
    beams[i, :] = np.matmul(A, V[:, i] * (1 + 0.007))  # scale factor 0.7%

beams_noise = beams + (0.042 ** 2) * np.random.randn(len(V[0, :]), 4) + \
              0.0001 * np.ones((len(V[0, :]), 4))

T = 3
Y = np.zeros((len(V[0, :]) - T, T, 4))
Z = np.zeros((len(V[0, :]) - T, 3))
X = np.zeros((len(V[0, :]) - T, 4))

V = V.T
for t in range(len(V[:, 0]) - T):
    x = beams_noise[t + T, :]
    X[t, :] = x
    y = beams_noise[t:t + T, :]
    Y[t, :] = y
    z = V[t + T, :]
    Z[t, :] = z

# Num of DVL samples
N = len(V[:, 0]) - T

# Make inputs and targets
X_test = torch.from_numpy(X[:, :].astype(np.float32))
X_test_DVL = torch.from_numpy(Y[:, :].astype(np.float32))
y_test = torch.from_numpy(Z[:, :].astype(np.float32))

s2 = X_test
sy = y_test

# Divide to batches
batch_size = 4

X_test = torch.utils.data.DataLoader(dataset=X_test, batch_size=batch_size)
X_test_DVL = torch.utils.data.DataLoader(dataset=X_test_DVL, batch_size=batch_size)
y_test = torch.utils.data.DataLoader(dataset=y_test, batch_size=batch_size)

# load the model
model = BeamsNetV2()
model.load_state_dict(torch.load('BeamsNetV2.pkl'))

# Move to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Results

model.eval()
with torch.no_grad():
    validation_predictions = []
    for inputs1, inputs2, targets in zip(X_test, X_test_DVL, y_test):
        inputs1, inputs2, targets = inputs1.to(device), inputs2.to(device), targets.to(device)
        validation_predictions.append(model(inputs1, inputs2).cpu().numpy())
validation_predictions = np.asarray(validation_predictions,dtype=object)
validation_predictions = np.concatenate(validation_predictions, axis=0)

predicted_v = np.zeros((N, 3))
LS_v = np.zeros((N, 3))
gt_v = np.zeros((N, 3))
for i in range(N):
    predicted_v[i, :] = validation_predictions[i, :]
    LS_v[i, :] = np.matmul(p_inv, transpose(s2[i, :]))
    gt_v[i, :] = sy[i, :]

# Rmse of the prediction
rmse_ls, rmse_predicted, improv = RMSE(gt_v, predicted_v, LS_v)
mae_ls, mae_predicted = MAE(gt_v, predicted_v, LS_v)
r2_ls, r2_predicted = NSE_R2(gt_v, predicted_v, LS_v)
vaf_ls, vaf_predicted = VAF(gt_v, predicted_v, LS_v)

df = pd.DataFrame(
    np.array([[rmse_predicted, mae_predicted, r2_predicted, vaf_predicted], [rmse_ls, mae_ls, r2_ls, vaf_ls]]),
    pd.Index([
        'Network prediction', 'Least Squares solution']), columns=['RMSE', 'MAE', 'NSE', 'VAF'])
print('BeamsNetV2 and Least Squares Results: ')
print(df)
