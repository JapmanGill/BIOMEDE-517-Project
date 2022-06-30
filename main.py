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

filename = "data/2021-04-12/batch_data_32ms_12chan_50bins_nonoverlapping.mat"
batch_data = scio.loadmat(filename)

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

dataset_train = FingerFlexionDataset(
    X_train, Y_train, x_train_0, X_train_mean, X_train_std, Y_train_mean, Y_train_std
)
dataset_val = FingerFlexionDataset(
    X_val, Y_val, x_val_0, X_train_mean, X_train_std, Y_train_mean, Y_train_std
)
train_dataloader = DataLoader(dataset_train, batch_size=4, shuffle=False)
val_dataloader = DataLoader(dataset_val)


modelFolder = "KNet/"
epochs = 150
n_batches = [64]
l_rates = [1e-3]
w_decays = [1e-5]
only_vels = [False]
for n_batch, w_decay, l_rate, only_vel in itertools.product(
    n_batches, w_decays, l_rates, only_vels
):
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
        "only_vel": only_vel,
        "CAR": False,
        "binsize": 32,
        "num_channels": 12,
        "seq_length": 50,
        "distinct_x0": True,
        "overlapping": False,
        "filename": filename,
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
        # pipeline.NNTrain(
        #     n_examples, train_dataloader, n_cv, val_dataloader, only_vel=only_vel
        # )
        pipeline.new_train(train_dataloader, val_dataloader, only_vel)
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
