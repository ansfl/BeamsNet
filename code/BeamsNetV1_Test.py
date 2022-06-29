################## Imports ##################

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from numpy import load
from numpy import transpose
from numpy import cos
from numpy import sin
from numpy.random.mtrand import random
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
from torch.utils.data import Dataset


# Seeds
torch.manual_seed(0)
np.random.seed(0)


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


################## DNN ##################
class BeamsNetV1(nn.Module):
    def __init__(self):
        super(BeamsNetV1, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=6,
                      kernel_size=2, stride=1),
            nn.Tanh(),
        )
        self.ConvToFc = nn.Sequential(
            nn.Linear(1188, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU(),
        )
        self.FC_output = nn.Sequential(
            nn.Linear(4 + 2, 3),
        )
        self.initialize_weights()

    def forward(self, x1, x2, y):
        x1 = self.conv_layer(x1)
        x2 = self.conv_layer(x2)
        # x = x.view(x.size(0), -1)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.column_stack((x1, x2))
        x = F.dropout(x, p=0.2)
        x = self.ConvToFc(x)
        x = torch.column_stack((x, y))
        x = self.FC_output(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)


################# MAIN ################

# load
path = os.getcwd()
path = os.path.abspath(os.path.join(path, os.pardir))
IMU_in = load(path + '\dataset\Test\IMU_in_test.npy')
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
    beams[i, :] = np.matmul(A, (V[:, i]) * (1 + 0.007))  # scale factor 0.7%

beams_noise = beams + (0.042 ** 2) * np.random.randn(len(V[0, :]), 4) + \
              0.001 * np.ones((len(V[0, :]), 4))

T = 100
X_gyro = np.zeros((len(IMU_in[0, :, 0]) // T, 3, T))
X_acc = np.zeros((len(IMU_in[0, :, 0]) // T, 3, T))
Y = np.zeros((len(IMU_in[0, :, 0]) // T, 4))
Z = np.zeros((len(IMU_in[0, :, 0]) // T, 3))

n = 0
V = V.T
for t in range(0, len(IMU_in[0, :, 0]) - 1, T):
    x_acc = IMU_in[:, t:t + T, 0]
    X_acc[n, :, :] = x_acc[:, :]
    x_gyro = IMU_in[:, t:t + T, 1]
    X_gyro[n, :, :] = x_gyro[:, :]
    y = beams_noise[n, :]
    Y[n, :] = y
    z = V[n, :]
    Z[n, :] = z
    n = n + 1

# Num of DVL samples
N = len(IMU_in[0, :, 0]) // T

# Make inputs and targets
X_test_acc = torch.from_numpy(X_acc[:, :, :].astype(np.float32))
X_test_gyro = torch.from_numpy(X_gyro[:, :, :].astype(np.float32))
X_test_DVL = torch.from_numpy(Y[:, :].astype(np.float32))
y_test = torch.from_numpy(Z[:, :].astype(np.float32))

s2 = X_test_DVL
sy = y_test

# Divide to batches
batch_size = 4
X_test_acc = torch.utils.data.DataLoader(dataset=X_test_acc, batch_size=batch_size)
X_test_gyro = torch.utils.data.DataLoader(dataset=X_test_gyro, batch_size=batch_size)
X_test_DVL = torch.utils.data.DataLoader(dataset=X_test_DVL, batch_size=batch_size)
y_test = torch.utils.data.DataLoader(dataset=y_test, batch_size=batch_size)

# load model
model = BeamsNetV1()
model.load_state_dict(torch.load('BeamsNetV1.pkl'))

# Move to GPU/CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()
with torch.no_grad():
    validation_predictions = []
    for inputs11, inputs12, inputs2, targets in zip(X_test_acc, X_test_gyro, X_test_DVL, y_test):
        inputs11, inputs12, inputs2, targets = inputs11.to(device), inputs12.to(device), inputs2.to(
            device), targets.to(device)
        validation_predictions.append(model(inputs11, inputs12, inputs2).cpu().numpy())
validation_predictions = np.asarray(validation_predictions, dtype=object)
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
print('BeamsNetV1 and Least Squares Results: ')
print(df)
