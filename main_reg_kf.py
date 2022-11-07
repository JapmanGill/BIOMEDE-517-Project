import sys
from datetime import datetime

import scipy.io as sio
import torch
import numpy as np

import wandb
from pybmi.utils import TrainingUtils

sys.path.append("kalmannet")

from kalman_net import KalmanNetNN
from pipeline_kf import Pipeline_KF

torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utils.utils import compute_correlation

# Load KF data

today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m_%d_%y")
strNow = now.strftime("%H_%M_%S")
strTime = strToday + "__" + strNow

# Fixed params
monkey = "Joker"
date = "2022-09-21"
run_train = "Run-002"
run_test = None  #'Run-003'
binsize = 32
fingers = [2, 4]
is_refit = False
train_test_split = 0.9
norm_x_movavg_bins = None
pred_type = "pv"
run_reg_kf = True
lrate = 0
wdecay = 0
batch_size = 1000
conv_size = 3
normalize_x = False
normalize_y = False

num_model = 2
kf_model = sio.loadmat(f"Z:/Data/Monkeys/{monkey}/{date}/decodeParamsKF{num_model}.mat")
good_chans_SBP = kf_model["chansSbp"]
good_chans_SBP_0idx = [x - 1 for x in good_chans_SBP][0]
num_states = (
    len(fingers) if pred_type == "v" else 2 * len(fingers)
)  # 2 if velocity only, 4 if pos+vel
# Include bias
num_states += 1

A = torch.tensor(kf_model["xpcA"])[:num_states, :num_states, 1]
C = torch.tensor(kf_model["xpcC"])[: len(good_chans_SBP_0idx), :num_states, 1]
W = torch.tensor(kf_model["xpcW"])[:num_states, :num_states, 1]
Q = torch.tensor(kf_model["Q"][0][1])

[loader_train, loader_val] = TrainingUtils.load_training_data(
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
    pred_type="pv",
    return_norm_params=False,
)
# sys_model.InitSequence(x_0, P_0)
knet_model = KalmanNetNN(binsize, reg_kf=run_reg_kf)
knet_model.build(A, C, W, Q)

# wandb.init(
#     project="kalman-net",
#     entity="lhcubillos",
#     group=f"optuna_{strTime}",
#     reinit=True,
# )

# Run the kalman filter
val_loss = torch.empty([len(loader_val), num_states - 1])
val_corr = np.zeros([len(loader_val), num_states - 1])
for j, loader_dict in enumerate(loader_val):
    y = loader_dict["chans_nohist"]
    # Remove bad channels
    y = torch.index_select(
        y, 1, torch.tensor(good_chans_SBP_0idx, dtype=torch.int).to(device)
    )
    x = loader_dict["states"]
    x = torch.cat([x, torch.ones(x.shape[0], 1).to(device)], 1)
    x_0 = loader_dict["initial_states"]
    x_0 = torch.cat([x_0, torch.ones(x_0.shape[0], 1).to(device)], 1)
    # x_0[:, [2, 3]] = x_0[:, [2, 3]] / binsize
    # Initialize hidden state: necessary for backprop
    # knet_model.init_hidden()
    # Initialize sequence for KalmanFilter
    knet_model.init_sequence(x_0[0, :])
    # Run model on validation set
    # Output is (seq_len, m)
    # Skip the first data point, as we used it to initialize the sequence
    x_hat = knet_model.forward_sequence(y[1:, :].T).T.to(device)
    # Compute MSE loss and correlation
    val_loss[j, :] = (
        ((x_hat[:, : (num_states - 1)] - x[1:, : (num_states - 1)]) ** 2)
        .mean(axis=[0])
        .detach()
    )
    val_corr[j, :] = compute_correlation(
        x[1:, : (num_states - 1)].detach().cpu().T,
        x_hat[:, : (num_states - 1)].detach().cpu().T,
    )

training_outputs = {
    "val_corr": val_corr,
}
training_inputs = {
    "pred_type": pred_type,
}
TrainingUtils.save_nn_decoder(
    monkey,
    date,
    knet_model,
    None,
    binsize,
    fingers,
    good_chans_SBP,
    training_inputs,
    training_outputs,
    fname_prefix="KNet",
)
print("hola")
