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

# from dataset import FingerFlexionDataset

from pybmi.utils import TrainingUtils

torch.set_default_dtype(torch.float32)

# filename = "data/2022-04-15_run11/batch_data_32ms_12chan_50bins_nonoverlapping_onlyvel_longval.mat"
# batch_data = scio.loadmat(filename)

# F = torch.tensor(batch_data["A"]).float()
# H = torch.tensor(batch_data["C"]).float()
# Q = torch.tensor(batch_data["W"]).float()
# R = torch.tensor(batch_data["Q"]).float()
# B = torch.tensor(batch_data["B"]).float()


# x_train_0 = torch.tensor(batch_data["bx_train_0"]).float()
# x_val_0 = torch.tensor(batch_data["bx_val_0"]).float()
# P_0 = Q
# X_test = torch.tensor(batch_data["X_test"]).t().float()
# Y_test = torch.tensor(batch_data["Y_test"]).t().float()
# X_train = torch.tensor(batch_data["bX_train"]).float()
# Y_train = torch.tensor(batch_data["bY_train"]).float()
# X_val = torch.tensor(batch_data["bX_val"]).float()
# Y_val = torch.tensor(batch_data["bY_val"]).float()
# X_train_mean = torch.tensor(batch_data["X_train_mean"]).float()
# X_train_std = torch.tensor(batch_data["X_train_std"]).float()
# Y_train_mean = torch.tensor(batch_data["Y_train_mean"]).float()
# Y_train_std = torch.tensor(batch_data["Y_train_std"]).float()

# # Add extra dimension
# X_test = X_test[None, :, :]
# Y_test = Y_test[None, :, :]
# x_test_0 = X_test[0, :, 0]
# x_test_0 = x_test_0[:, None]
# x_train_0 = x_train_0[:, :, None]
# x_val_0 = x_val_0[:, :, None]

# T = X_train.size()[2]
# T_val = X_val.size()[2]
# T_test = X_test.size()[2]

# n_examples = X_train.size()[0]
# n_cv = X_val.size()[0]
# n_test = X_test.size()[0]

# dataset_train = FingerFlexionDataset(
#     X_train, Y_train, x_train_0, X_train_mean, X_train_std, Y_train_mean, Y_train_std
# )
# dataset_val = FingerFlexionDataset(
#     X_val, Y_val, x_val_0, X_train_mean, X_train_std, Y_train_mean, Y_train_std
# )

monkey = "Joker"
date = "2022-09-21"
run_train = "Run-002"
run_test = None  #'Run-003'
binsize = 32
fingers = [2, 4]
normalize_x = True
normalize_y = False
conv_size = 40
is_refit = False
norm_x_movavg_bins = None
batch_size = 16
train_test_split = 0.8
pred_type = "pv"

# Load KF data
import scipy.io as sio

num_model = 2
kf_model = sio.loadmat(f"Z:/Data/Monkeys/{monkey}/{date}/decodeParamsKF{num_model}.mat")
good_chans_SBP = kf_model["chansSbp"]
good_chans_SBP_0idx = [x - 1 for x in good_chans_SBP][0]
num_states = (
    len(fingers) if pred_type == "v" else 2 * len(fingers)
)  # 2 if velocity only, 4 if pos+vel

# if pred_type == "v":
#     A = torch.tensor(kf_model["xpcA"])[2:4, 2:4, 1]
#     C = torch.tensor(kf_model["xpcC"])[: len(good_chans_SBP_0idx), 2:5, 1]
# elif pred_type == "pv":
A = torch.tensor(kf_model["xpcA"])[:num_states, :num_states, 1]
C = torch.tensor(kf_model["xpcC"])[: len(good_chans_SBP_0idx), : num_states + 1, 1]
# else:
#     raise NotImplementedError

[loader_train, loader_val, neural_mean, neural_std] = TrainingUtils.load_training_data(
    monkey,
    date,
    run_train,
    run_test=run_test,
    good_chans_0idx=good_chans_SBP_0idx,
    isrefit=is_refit,
    fingers=fingers,
    binsize=binsize,
    batch_size=batch_size,
    binshist=conv_size,
    normalize_x=normalize_x,
    normalize_y=normalize_y,
    norm_x_movavg_bins=norm_x_movavg_bins,
    train_test_split=train_test_split,  # only used if run_test is None
    pred_type=pred_type,
    return_norm_params=True,
)

modelFolder = "KNet/"
epochs = 80
n_batches = [16]
l_rates = [1e-3]
w_decays = [0]
zero_hidden_states = [True]
non_linears = [False]
for n_batch, w_decay, l_rate, zero_hidden_state, non_linear in itertools.product(
    n_batches, w_decays, l_rates, zero_hidden_states, non_linears
):
    print(n_batch, l_rate, w_decay)
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m_%d_%y")
    strNow = now.strftime("%H_%M_%S")
    strTime = strToday + "__" + strNow

    # train_dataloader = DataLoader(dataset_train, batch_size=n_batch, shuffle=False)
    # val_dataloader = DataLoader(dataset_val)

    # Start run
    config = {
        "epochs": epochs,
        "l_rate": l_rate,
        "n_batch": n_batch,
        "w_decay": w_decay,
        "only_vel": False,
        "CAR": False,
        "binsize": 32,
        "num_channels": 12,
        "seq_length": conv_size,
        "distinct_x0": True,
        "overlapping": True,  #        "filename": filename,
        "normalizing": True,
        "zero_hidden_state": zero_hidden_state,
        "non_linear": non_linear,
    }
    run = wandb.init(
        project="kalman-net",
        entity="lhcubillos",
        reinit=True,
        name=f"fingflexion_{strTime}",
        config=config,
    )

    pipeline = Pipeline_KF(
        "models",
        f"KNet_fingflexion_{strTime}",
        good_chans_SBP_0idx,
        pred_type="v",
    )
    sys_model = SystemModel(A, None, C, None, 0, 0, None)
    # sys_model.InitSequence(x_0, P_0)
    pipeline.setssModel(sys_model)
    KNet_model = KalmanNetNN(zero_hidden_state, non_linear)
    KNet_model.Build(A, C)
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
        pipeline.new_train(loader_train, loader_val)
    # pipeline.NNTest(n_test, Y_test, X_test, x_test_0)
    # except Exception as e:
    #     print(f"[Error]: {e}")
    finally:
        #     print("saving pipeline...")
        #     pipeline.save()
        #     del pipeline
        #     del sys_model
        #     del KNet_model
        run.finish()
