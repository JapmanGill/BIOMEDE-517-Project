import wandb
import os
import sys

sys.path.append("kalmannet")
from Pipeline_KF import Pipeline_KF
from KalmanNet import KalmanNetNN
from Linear_sysmdl import SystemModel
import torch
import scipy.io as scio
import numpy as np
from datetime import datetime
import itertools

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

batch_data = scio.loadmat("data/finger_flexion/batch_data_32ms_15chan_100bins.mat")

F = torch.tensor(matA["A"]).float()
H = torch.tensor(batch_data["C"]).float()
Q = torch.tensor(matW["W"]).float()
R = torch.tensor(matQ["Q"]).float()
# x_test_0 = torch.tensor(batch_data["bx_test_0"]).t().float()
x_train_0 = torch.tensor(batch_data["bx_train_0"]).float()
x_val_0 = torch.tensor(batch_data["bx_val_0"]).float()
P_0 = Q
X_test = torch.tensor(batch_data["X_test"]).t().float()
Y_test = torch.tensor(batch_data["Y_test"]).t().float()
X_train = torch.tensor(batch_data["bX_train"]).float()
Y_train = torch.tensor(batch_data["bY_train"]).float()
X_val = torch.tensor(batch_data["bX_val"]).float()
Y_val = torch.tensor(batch_data["bY_val"]).float()

# Add extra dimension
X_test = X_test[None, :, :]
Y_test = Y_test[None, :, :]
x_test_0 = X_test[0, :, 0]
x_test_0 = x_test_0[:, None]
x_train_0 = x_train_0[:, :, None]
x_val_0 = x_val_0[:, :, None]

T = X_train.size()[2]
T_val = X_val.size()[2]
T_test = X_test.size()[2]

n_examples = X_train.size()[0]
n_cv = X_val.size()[0]
n_test = X_test.size()[0]


modelFolder = "KNet/"
epochs = 2
n_batches = [64, 128]
l_rates = [1e-3, 1e-4]
w_decays = [1e-3, 1e-4]
for n_batch, w_decay, l_rate in itertools.product(n_batches, w_decays, l_rates):
    print(n_batch, l_rate, w_decay)
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m_%d_%y")
    strNow = now.strftime("%H_%M_%S")
    strTime = strToday + "__" + strNow

    # Start run
    config = {
        "epochs": epochs,
        "l_rate": l_rate,
        "n_batch": n_batch,
        "w_decay": w_decay,
        "CAR": False,
        "binsize": 32,
        "num_channels": 15,
        "seq_length": 100,
        "distinct_x0": True,
    }
    run = wandb.init(
        project="kalman-net",
        entity="lhcubillos",
        reinit=True,
        name=f"fingflexion_{strTime}",
        config=config,
    )

    pipeline = Pipeline_KF(strTime, "models", f"KNet_fingflexion_{strTime}")
    sys_model = SystemModel(F, Q, H, R, T, T_val, T_test)
    sys_model.InitSequence(x_0, P_0)
    pipeline.setssModel(sys_model)
    KNet_model = KalmanNetNN()
    KNet_model.Build(sys_model)
    pipeline.setModel(KNet_model)
    pipeline.setTrainingParams(
        n_Epochs=epochs,
        n_Batch=n_batch,
        learningRate=l_rate,
        weightDecay=w_decay,
    )

    try:
        pass
        pipeline.NNTrain(
            n_examples, Y_train, X_train, x_train_0, n_cv, Y_val, X_val, x_val_0
        )
        pipeline.NNTest(n_test, Y_test, X_test, x_test_0)
    except Exception as e:
        print(f"[Error]: {e}")
    finally:
        pipeline.save()
        del pipeline
        del sys_model
        del KNet_model
        run.finish()
