import sys
from datetime import datetime

import joblib
import scipy.io as sio
import torch

import optuna
import wandb
from pybmi.utils import TrainingUtils

sys.path.append("kalmannet")

from kalman_net import KalmanNetNN
from pipeline_kf import Pipeline_KF

torch.set_default_dtype(torch.float32)

# Load KF data

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
    num_states += 1

    # FIXME: figure out what to do so that the data is not normalized but the network receives normalized data
    A = torch.tensor(kf_model["xpcA"])[:num_states, :num_states, 1]
    C = torch.tensor(kf_model["xpcC"])[: len(good_chans_SBP_0idx), :num_states, 1]
    m = A.size()[0]
    n = C.size()[0]
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
        batch_size=trial.suggest_int("batch_size", 4, 64, step=4),
        binshist=trial.suggest_int("conv_size", 20, 80),
        normalize_x=trial.suggest_categorical("normalize_x", [True, False]),
        normalize_y=False,
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
    # sys_model.InitSequence(x_0, P_0)
    KNet_model = KalmanNetNN(
        h1_size=trial.suggest_int(
            "h1_size", (m + n) * (10) * 1, (m + n) * (10) * 10, step=(m + n) * (10)
        ),
        h2_size=trial.suggest_int(
            "h2_size", (m * n) * (4), (m * n) * (4) * 10, step=(m * n)
        ),
        hidden_dim=trial.suggest_int(
            "hidden_dim",
            (m * m + n * n) * 1,
            (m * m + n * n) * 10,
            step=(m * m + n * n),
        ),
        gain_scaler=trial.suggest_float("gain_scaler", 5e3, 5e4, log=True),
    )
    KNet_model.build(A, C)
    pipeline.set_model(KNet_model)
    pipeline.set_training_params(
        n_epochs=20,
        learning_rate=trial.suggest_float("l_rate", 1e-6, 1e-3, log=True),
        weight_decay=trial.suggest_float("w_decay", 1e-6, 1e-4, log=True),
    )

    config = dict(trial.params)
    config["trial.number"] = trial.number
    wandb.init(
        project="kalman-net",
        entity="lhcubillos",
        group=f"optuna_{strTime}",
        config=config,
        reinit=True,
    )
    val_loss = pipeline.train(
        loader_train,
        loader_val,
        compute_val_every=20,
        stop_at_iterations=300,
        trial=trial,
    )
    return val_loss
    # except Exception as e:
    #     print(f"[Error]: {e}")
    # finally:
    #     print("saving pipeline...")
    #     pipeline.save()
    #     del pipeline
    #     del sys_model
    #     del KNet_model
    # run.finish()


study = optuna.create_study()
try:
    study.optimize(train_kalmannet, n_trials=100)
finally:
    joblib.dump(study, f"optuna/study_{strTime}.pkl")
