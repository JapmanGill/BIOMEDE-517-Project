import os

dir = os.path.dirname(os.path.realpath("main_ekf.py"))
import sys

sys.path.append(dir + "/")
sys.path.append(dir + "/" + "kalmannet")
from Pipeline_KF_bme import Pipeline_KF
from KalmanNet_nn_arch1 import KalmanNetNN
from Linear_sysmdl_bme import SystemModel
import torch
import scipy.io as scio
import numpy as np
from datetime import datetime

matA = scio.loadmat("data/contdata/A.mat")
matC = scio.loadmat("data/contdata/C.mat")
matQ = scio.loadmat("data/contdata/Q.mat")
matW = scio.loadmat("data/contdata/W.mat")
x_0 = scio.loadmat("data/contdata/x_0.mat")
X_test = scio.loadmat("data/contdata/X_test.mat")
Y_test = scio.loadmat("data/contdata/Y_test.mat")
X_train = scio.loadmat("data/contdata/X_train.mat")
Y_train = scio.loadmat("data/contdata/Y_train.mat")
X_val = scio.loadmat("data/contdata/X_val.mat")
Y_val = scio.loadmat("data/contdata/Y_val.mat")

F = torch.tensor(matA["A"]).float()
H = torch.tensor(matC["C"]).float()
Q = torch.tensor(matW["W"]).float()
R = torch.tensor(matQ["Q"]).float()
x_0 = torch.tensor(x_0["x_0"]).t().float()
P_0 = Q
X_test = torch.tensor(X_test["X_test"]).t().float()
Y_test = torch.tensor(Y_test["Y_test"]).t().float()
X_train = torch.tensor(X_train["X_train"]).t().float()
Y_train = torch.tensor(Y_train["Y_train"]).t().float()
X_val = torch.tensor(X_val["X_val"]).t().float()
Y_val = torch.tensor(Y_val["Y_val"]).t().float()
T = X_train.size()[1]
T_val = X_val.size()[1]
T_test = X_test.size()[1]

# Add extra dimension
X_test = X_test[None, :, :]
Y_test = Y_test[None, :, :]
X_train = X_train[None, :, :]
Y_train = Y_train[None, :, :]
X_val = X_val[None, :, :]
Y_val = Y_val[None, :, :]


today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow

print("Start KNet pipeline")
modelFolder = "KNet" + "/"
KNet_Pipeline = Pipeline_KF(strTime, "models", "KNet_Linear_Neural")
sys_model = SystemModel(F, Q, H, R, T, T_val, T_test)
sys_model.InitSequence(x_0, P_0)
KNet_Pipeline.setssModel(sys_model)
# KNet_model = KalmanNetNN()
# KNet_model.Build(sys_model)
KNet_model = torch.load("models/model_KNet_Linear_Neural")
KNet_model.InitSystemDynamics(F, H)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(
    n_Epochs=500, n_Batch=1, learningRate=1e-3, weightDecay=1e-5
)

KNet_Pipeline.NNTrain(1, Y_train, X_train, 1, Y_val, X_val)
[
    KNet_MSE_test_linear_arr,
    KNet_MSE_test_linear_avg,
    KNet_MSE_test_dB_avg,
    KNet_test,
] = KNet_Pipeline.NNTest(1, Y_test, X_test)
# KNet_Pipeline.save()
