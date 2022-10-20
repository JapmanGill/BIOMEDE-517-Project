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

import optuna
import joblib

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

# Load KF data
import scipy.io as sio

# num_model = 2
# kf_model = sio.loadmat(f"Z:/Data/Monkeys/{monkey}/{date}/decodeParamsKF{num_model}.mat")
# good_chans_SBP = kf_model["chansSbp"]
# good_chans_SBP_0idx = [x - 1 for x in good_chans_SBP][0]
# num_states = (
#     len(fingers) if pred_type == "v" else 2 * len(fingers)
# )  # 2 if velocity only, 4 if pos+vel

# if pred_type == "v":
#     A = torch.tensor(kf_model["xpcA"])[2:4, 2:4, 1]
#     C = torch.tensor(kf_model["xpcC"])[: len(good_chans_SBP_0idx), 2:5, 1]
# elif pred_type == "pv":
# A = torch.tensor(kf_model["xpcA"])[:num_states, :num_states, 1]
# C = torch.tensor(kf_model["xpcC"])[: len(good_chans_SBP_0idx), : num_states + 1, 1]
# else:
#     raise NotImplementedError

# modelFolder = "KNet/"
# epochs = 80
# n_batches = [16]
# l_rates = [1e-3]
# w_decays = [0]
# zero_hidden_states = [True]
# non_linears = [False]

today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m_%d_%y")
strNow = now.strftime("%H_%M_%S")
strTime = strToday + "__" + strNow


def train_kalmannet(trial):
    # Fixed params
    monkey = "Joker"
    date = "2022-09-21"
    run_train = "Run-002"
    run_test = None  #'Run-003'
    binsize = 32
    fingers = [2, 4]
    is_refit = False
    train_test_split = 0.8
    norm_x_movavg_bins = None
    pred_type = "pv"

    # WandB

    num_model = 2
    kf_model = sio.loadmat(
        f"Z:/Data/Monkeys/{monkey}/{date}/decodeParamsKF{num_model}.mat"
    )
    good_chans_SBP = kf_model["chansSbp"]
    good_chans_SBP_0idx = [x - 1 for x in good_chans_SBP][0]
    num_states = (
        len(fingers) if pred_type == "v" else 2 * len(fingers)
    )  # 2 if velocity only, 4 if pos+vel

    A = torch.tensor(kf_model["xpcA"])[:num_states, :num_states, 1]
    C = torch.tensor(kf_model["xpcC"])[: len(good_chans_SBP_0idx), : num_states + 1, 1]

    [
        loader_train,
        loader_val,
        neural_mean,
        neural_std,
    ] = TrainingUtils.load_training_data(
        monkey,
        date,
        run_train,
        run_test=run_test,
        good_chans_0idx=good_chans_SBP_0idx,
        isrefit=is_refit,
        fingers=fingers,
        binsize=binsize,
        batch_size=trial.suggest_int("batch_size", 4, 32, step=4),
        binshist=trial.suggest_int("conv_size", 20, 32, step=4),
        normalize_x=trial.suggest_categorical("normalize_x", [True, False]),
        normalize_y=trial.suggest_categorical("normalize_y", [True, False]),
        norm_x_movavg_bins=norm_x_movavg_bins,
        train_test_split=train_test_split,  # only used if run_test is None
        pred_type="pv",
        return_norm_params=True,
    )
    pipeline = Pipeline_KF(
        "models",
        f"KNet_fingflexion_{strTime}",
        good_chans_SBP_0idx,
        pred_type=trial.suggest_categorical("pred_type", ["pv", "v"]),
    )
    sys_model = SystemModel(A, None, C, None, 0, 0, None)
    # sys_model.InitSequence(x_0, P_0)
    pipeline.setssModel(sys_model)
    KNet_model = KalmanNetNN()
    KNet_model.Build(A, C)
    pipeline.setModel(KNet_model)
    pipeline.setTrainingParams(
        n_Epochs=20,
        learningRate=trial.suggest_float("l_rate", 1e-5, 1e-3, log=True),
        weightDecay=trial.suggest_float("w_decay", 1e-6, 1e-4, log=True),
    )

    # try:
    # pipeline.NNTrain(
    #     n_examples, train_dataloader, n_cv, val_dataloader, only_vel=only_vel
    # )
    config = dict(trial.params)
    config["trial.number"] = trial.number
    wandb.init(
        project="kalman-net",
        entity="lhcubillos",
        group=f"optuna_{strTime}",
        config=config,
        reinit=True,
    )
    val_loss = pipeline.new_train(
        loader_train,
        loader_val,
        compute_val_every=20,
        stop_at_iterations=200,
        trial=trial,
    )
    return val_loss
    # pipeline.NNTest(n_test, Y_test, X_test, x_test_0)
    # except Exception as e:
    #     print(f"[Error]: {e}")
    # finally:
    #     print("saving pipeline...")
    #     pipeline.save()
    #     del pipeline
    #     del sys_model
    #     del KNet_model
    # run.finish()


today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m_%d_%y")
strNow = now.strftime("%H_%M_%S")
strTime = strToday + "__" + strNow

study = optuna.create_study()
try:
    study.optimize(train_kalmannet, n_trials=100)
finally:
    joblib.dump(study, f"optuna/study_{strTime}.pkl")
