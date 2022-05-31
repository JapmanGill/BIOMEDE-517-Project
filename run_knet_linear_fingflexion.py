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

matA = scio.loadmat("data/finger_flexion/A.mat")
matC = scio.loadmat("data/finger_flexion/C.mat")
matQ = scio.loadmat("data/finger_flexion/Q.mat")
matW = scio.loadmat("data/finger_flexion/W.mat")
x_0 = scio.loadmat("data/finger_flexion/x_0.mat")
X_test = scio.loadmat("data/finger_flexion/X_test.mat")
Y_test = scio.loadmat("data/finger_flexion/Y_test.mat")
X_train = scio.loadmat("data/finger_flexion/X_train.mat")
Y_train = scio.loadmat("data/finger_flexion/Y_train.mat")
X_val = scio.loadmat("data/finger_flexion/X_val.mat")
Y_val = scio.loadmat("data/finger_flexion/Y_val.mat")

batch_data = scio.loadmat("data/finger_flexion/batch_data_32ms_20chan.mat")

F = torch.tensor(matA["A"]).float()
H = torch.tensor(batch_data["C"]).float()
Q = torch.tensor(matW["W"]).float()
R = torch.tensor(matQ["Q"]).float()
# x_0 = torch.tensor(batch_data["bx_0"]).t().float()
x_0 = torch.tensor([[0.5498], [0.5771], [-0.0003], [-0.0001]])
P_0 = Q
X_test = torch.tensor(X_test["X_test"]).t().float()
Y_test = torch.tensor(Y_test["Y_test"]).t().float()
X_train = torch.tensor(batch_data["bX_train"]).float()
Y_train = torch.tensor(batch_data["bY_train"]).float()
X_val = torch.tensor(batch_data["bX_val"]).float()
Y_val = torch.tensor(batch_data["bY_val"]).float()

# Add extra dimension
X_test = X_test[None, :, :]
Y_test = Y_test[None, :, :]
# X_train = X_train[None, :, :]
# Y_train = Y_train[None, :, :]
# X_val = X_val[None, :, :]
# Y_val = Y_val[None, :, :]

T = X_train.size()[2]
T_val = X_val.size()[2]
T_test = X_test.size()[2]

n_examples = X_train.size()[0]
n_cv = X_val.size()[0]
n_test = X_test.size()[0]

# # Add extra dimension
# X_test = X_test[None, :, :]
# Y_test = Y_test[None, :, :]
# X_train = X_train[None, :, :]
# Y_train = Y_train[None, :, :]
# X_val = X_val[None, :, :]
# Y_val = Y_val[None, :, :]


today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m_%d_%y")
strNow = now.strftime("%H_%M_%S")
strTime = strToday + "__" + strNow

print("Start KNet pipeline")
modelFolder = "KNet" + "/"
KNet_Pipeline = Pipeline_KF(strTime, "models", f"KNet_fingflexion_{strTime}")
sys_model = SystemModel(F, Q, H, R, T, T_val, T_test)
sys_model.InitSequence(x_0, P_0)
KNet_Pipeline.setssModel(sys_model)
KNet_model = KalmanNetNN()
KNet_model.Build(sys_model)
# KNet_model = torch.load("models/model_KNet_Linear_Neural")
# KNet_model.InitSystemDynamics(F, H)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(
    n_Epochs=500, n_Batch=64, learningRate=1e-3, weightDecay=1e-4
)

KNet_Pipeline.NNTrain(n_examples, Y_train, X_train, n_cv, Y_val, X_val)
[
    KNet_MSE_test_linear_arr,
    KNet_MSE_test_linear_avg,
    KNet_MSE_test_dB_avg,
    KNet_test,
] = KNet_Pipeline.NNTest(n_test, Y_test, X_test)
KNet_Pipeline.save()
