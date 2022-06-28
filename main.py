import wandb
import os
import sys

sys.path.append("kalmannet")
from Pipeline_KF import Pipeline_KF
from KalmanNet import KalmanNetNN
from Linear_sysmdl import SystemModel
import torch
from torch.utils.data import DataLoader
import scipy.io as scio
import numpy as np
from datetime import datetime
import itertools
import traceback

from dataset import FingerFlexionDataset

batch_data = scio.loadmat(
    "data/finger_flexion/batch_data_32ms_20chan_5bins_overlapping.mat"
)

matA = scio.loadmat("data/finger_flexion/A.mat")
matQ = scio.loadmat("data/finger_flexion/Q.mat")
matW = scio.loadmat("data/finger_flexion/W.mat")

F = torch.tensor(batch_data["A"]).float()
H = torch.tensor(batch_data["C"]).float()
Q = torch.tensor(batch_data["W"]).float()
R = torch.tensor(batch_data["Q"]).float()

x_train_0 = torch.tensor(batch_data["bx_train_0"]).float()
x_val_0 = torch.tensor(batch_data["bx_val_0"]).float()
P_0 = Q
X_test = torch.tensor(batch_data["X_test"]).t().float()
Y_test = torch.tensor(batch_data["Y_test"]).t().float()
X_train = torch.tensor(batch_data["bX_train"]).float()
Y_train = torch.tensor(batch_data["bY_train"]).float()
X_val = torch.tensor(batch_data["bX_val"]).float()
Y_val = torch.tensor(batch_data["bY_val"]).float()
X_train_mean = torch.tensor(batch_data["X_train_mean"]).float()
X_train_std = torch.tensor(batch_data["X_train_std"]).float()
Y_train_mean = torch.tensor(batch_data["Y_train_mean"]).float()
Y_train_std = torch.tensor(batch_data["Y_train_std"]).float()

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

# dataset_train = FingerFlexionDataset(X_train, Y_train, x_train_0)
# dataset_val = FingerFlexionDataset(
#     X_val, Y_val, x_val_0, X_train_mean, X_train_std, Y_train_mean, Y_train_std
# )
# train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(dataset_val)


modelFolder = "KNet/"
epochs = 500
# n_batches = [64, 128]
# l_rates = [1e-3, 1e-4]
# w_decays = [1e-4, 1e-5]
n_batches = [64]
l_rates = [1e-4]
w_decays = [1e-5]
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
    # sys_model.InitSequence(x_0, P_0)
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
        pipeline.NNTrain(
            n_examples, Y_train, X_train, x_train_0, n_cv, Y_val, X_val, x_val_0
        )
        # pipeline.NNTest(n_test, Y_test, X_test, x_test_0)
    except Exception as e:
        print(f"[Error]: {e}")
    finally:
        print("saving pipeline...")
        pipeline.save()
        del pipeline
        del sys_model
        del KNet_model
        run.finish()
