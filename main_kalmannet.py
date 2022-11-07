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
train_test_split = 0.8
norm_x_movavg_bins = None
pred_type = "pv"
run_reg_kf = True
lrate = 0.0002973485
wdecay = 0.0000172
batch_size = 48
conv_size = 70
normalize_x = False
normalize_y = False
h1_size = 510
h2_size = 1560
hidden_dim = 845

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
knet_model = KalmanNetNN(
    binsize, reg_kf=run_reg_kf, h1_size=h1_size, h2_size=h2_size, hidden_dim=hidden_dim
)
knet_model.build(A, C)
pipeline = Pipeline_KF(
    "models",
    f"KNet_fingflexion_{strTime}",
    good_chans_SBP_0idx,
    pred_type="pv",
)
# sys_model.InitSequence(x_0, P_0)
KNet_model = KalmanNetNN(binsize)
KNet_model.build(A, C)
pipeline.set_model(KNet_model)
pipeline.set_training_params(
    n_epochs=2,
    learning_rate=1e-3,
    weight_decay=0,
)
# wandb.init(
#     project="kalman-net",
#     entity="lhcubillos",
#     name=f"test_{strTime}",
#     config={},
# )
val_loss = pipeline.train(
    loader_train, loader_val, compute_val_every=10, stop_at_iterations=5
)
torch.save(KNet_model, f"models/KNet_fingflexion_{strTime}.mdl")
training_outputs = {
    "val_loss": val_loss,
}
training_inputs = {
    "pred_type": pred_type,
}
TrainingUtils.save_nn_decoder(
    monkey,
    date,
    KNet_model,
    None,
    binsize,
    fingers,
    good_chans_SBP,
    training_inputs,
    training_outputs,
    fname_prefix="KNet",
)
print("hola")
