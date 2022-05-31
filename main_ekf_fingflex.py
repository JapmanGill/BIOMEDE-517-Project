#%%
import os

dir = os.path.dirname(os.path.realpath("main_ekf.py"))
import sys

sys.path.append(dir + "/")
sys.path.append(dir + "/" + "kalmannet")
from model_ekf import f, h
from parameters_ekf_fingflex import Q, W
from Extended_sysmdl import SystemModel
from Pipeline_EKF import Pipeline_EKF
from Extended_KalmanNet_nn import KalmanNetNN
import torch
import scipy.io as scio
import numpy as np
from datetime import datetime

#%%
# Get Time
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow

#%%
# Load Data
# Initial State
x_0 = scio.loadmat("data/finger_flexion/x_0.mat")
# Test Set
X_test = scio.loadmat("data/finger_flexion/X_test.mat")
Y_test = scio.loadmat("data/finger_flexion/Y_test.mat")
# Training Set
X_train = scio.loadmat("data/finger_flexion/X_train.mat")
Y_train = scio.loadmat("data/finger_flexion/Y_train.mat")
# Validation Set
X_val = scio.loadmat("data/finger_flexion/X_val.mat")
Y_val = scio.loadmat("data/finger_flexion/Y_val.mat")

# Converting to Tensors
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
m = X_train.shape[0]
n = Y_train.shape[0]

#%%
# Add extra dimension
X_test = X_test[None, :, :]
Y_test = Y_test[None, :, :]
X_train = X_train[None, :, :]
Y_train = Y_train[None, :, :]
X_val = X_val[None, :, :]
Y_val = Y_val[None, :, :]

#%%
# Training Pipeline
print("Start KNet pipeline")

# Preparing System Model
sys_model = SystemModel(f, Q, h, W, T, T_val, T_test, m, n, "EKF_fingflex")
sys_model.InitSequence(x_0, P_0)

# Creating Training Pipeline
KNet_Pipeline = Pipeline_EKF(strTime, "models", "EKF_fingflex")
KNet_Pipeline.setssModel(sys_model)
KNet_model = KalmanNetNN()
KNet_model.Build(sys_model)
KNet_Pipeline.setModel(KNet_model)

# Define Training Parameters
KNet_Pipeline.setTrainingParams(
    n_Epochs=2, n_Batch=1, learningRate=1e-3, weightDecay=1e-5
)

# ToDo: Check the Arguments for Testing & Training
KNet_Pipeline.NNTrain(1, Y_train, X_train, 1, Y_val, X_val)

######### ToDo: Check how Test Scores are being calculated
[
    KNet_MSE_test_linear_arr,
    KNet_MSE_test_linear_avg,
    KNet_MSE_test_dB_avg,
    KNet_test,
] = KNet_Pipeline.NNTest(1, Y_test, X_test)
KNet_Pipeline.save()
