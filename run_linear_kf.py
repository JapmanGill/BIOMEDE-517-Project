import sys

sys.path.append("G:/My Drive/UMich/Courses/W22/BME 517/KalmanNet_TSP")

from Linear_sysmdl import SystemModel
from Linear_KF_bme import KalmanFilter
import torch
import scipy.io as scio
import numpy as np

matA = scio.loadmat("BME517/data/F_mat_50_kf.mat")
matC = scio.loadmat("BME517/data/H_mat_50_kf.mat")
matQ = scio.loadmat("BME517/data/Q_mat_50_kf.mat")
matW = scio.loadmat("BME517/data/W_mat_50_kf.mat")
x_0 = scio.loadmat("BME517/data/x_0_50_kf.mat")
X_test = scio.loadmat("BME517/data/Xtest_50_kf.mat")
Y_test = scio.loadmat("BME517/data/Ytest_50_kf.mat")

F = torch.tensor(matA["A"])
H = torch.tensor(matC["C"])
Q = torch.tensor(matW["W"])
R = torch.tensor(matQ["Q"])
x_0 = torch.tensor(x_0["x_0"]).t()
P_0 = Q
X_test = torch.tensor(X_test["X_test"])
Y_test = torch.tensor(Y_test["Y_test"]).t()
T = X_test.size()[0]

kf = KalmanFilter(F, Q, H, R, T)
kf.InitSequence(x_0, P_0)
kf.GenerateSequence(Y_test, T)

corr = np.zeros(4)
for i in range(4):
    corr[i] = np.corrcoef(kf.x[i, :].cpu(), X_test[:, i].cpu())[0, 1]
mse_x = np.square(np.subtract(kf.x.cpu(), X_test.t().cpu())).mean(1)

print(f"Correlation: {corr}, MSE: {mse_x}")
