import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torch.optim as optim
import scipy.io as sio
import numpy as np
import numpy.matlib
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sys import platform
import pickle
import os
import warnings
import copy
from scipy.spatial import KDTree
from scipy import signal
import pybmi
from pybmi.utils.ZTools import ZStructTranslator, getZFeats, zarray
from pybmi.utils import ZTools
from scipy.signal import find_peaks

dtype = torch.float
ltype = torch.long


class OnlineDatasets(Dataset):
    """Offline dataset"""

    def __init__(
        self,
        mat_file,
        root_dir,
        transform=None,
        zero_train=False,
        Resamp=False,
        predtype="v",
        numfingers=2,
        numdelays=3,
        last_timestep_recent=False,
    ):
        """
        Args:
            mat_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with data files.
            transform (callable, optional): Optional transform to be applied on a sample.
            zero_train (bool, optional): TODO: figure out
            Resamp (bool, optional): TODO: figure out
            numdelays (int, optional): Number of delay bins. Defaults to 3.
            last_timestep_recent (bool, optional): If the last timestep should be the most recent data (used in RNNs)
        """
        mat = sio.loadmat(root_dir + mat_file)
        Xtrain_temp = torch.tensor(mat["X"]).to(device).to(dtype)
        Ytrain_temp = torch.tensor(mat["Y"]).to(device).to(dtype)
        mask = torch.tensor(mat["mask"])[:, 0].to(device).to(dtype)

        Xtrain_temp = mask * Xtrain_temp

        Xtrain1 = torch.zeros(
            (int(Xtrain_temp.shape[0]), int(Xtrain_temp.shape[1]), numdelays),
            device=device,
            dtype=dtype,
        )
        Xtrain1[:, :, 0] = Xtrain_temp
        for k1 in range(numdelays - 1):
            k = k1 + 1
            Xtrain1[k:, :, k] = Xtrain_temp[0:-k, :]

        if last_timestep_recent:
            # for RNNs, we want the last timestep to be the most recent data
            Xtrain1 = torch.flip(Xtrain1, (2,))

        if predtype == "v":
            Ytrain1 = Ytrain_temp[:, numfingers : numfingers * 2]
        if predtype == "p":
            Ytrain1 = Ytrain_temp[:, 0:numfingers]
        if predtype == "pv":
            Ytrain1 = Ytrain_temp[:, 0 : numfingers * 2]
        else:
            RuntimeError(
                "Must specify prediction type as Position 'p', Velocity 'v', or both 'pv'"
            )
        ind1 = mat["ind"].astype(int)

        if Resamp:
            ind1 = mat["ind"].astype(int)
            Xtrain = Xtrain1[ind1[:, 0] - 1, :, :]
            Ytrain = Ytrain1[ind1[:, 0] - 1, :]

        elif zero_train:
            ind = torch.sqrt(torch.sum(Ytrain1**2, 1)) < 0.001
            Xtrain = Xtrain1[ind, :, :]
            Ytrain = Ytrain1[ind, :]

        else:
            Xtrain = Xtrain1
            Ytrain = Ytrain1

        self.chan_states = (Xtrain, Ytrain)
        self.root_dir = root_dir
        self.transform = transform
        self.train = True

    def __len__(self):
        return len(self.chan_states[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        chans = self.chan_states[0][idx, :]
        states = self.chan_states[1][idx, :]

        sample = {"states": states, "chans": chans}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FingerDataset(Dataset):
    """Torch Dataset for predicting finger position/velocity from neural data.
    Nearly identical to the older 'OnlineDatasets', but here we pass in previously-loaded data
    """

    def __init__(
        self,
        X_neural,
        Y_fings,
        transform=None,
        zero_train=False,
        Resamp=False,
        predtype="v",
        numfingers=2,
        numdelays=3,
        positioninput=False,
        last_timestep_recent=True,
    ):
        """
        Args:
            X_neural (ndarray): Neural data, [n, neu], where neu is the number of channels
            Y_neural (ndarray): Behavioral data, [n, dim], where dim is the number of behavioral states
            transform (callable, optional): Optional transform to be applied on a sample.
            zero_train (bool, optional): TODO: figure out
            Resamp (bool, optional): TODO: figure out
            predtype ('v', 'p', or 'pv'): selects if position, velocity, or both are in the 'y' output
            numdelays (int, optional): Number of delay bins. Defaults to 3.
            positioninput (bool, optional): If True, the previous position is appended to the neural data as additional features
            last_timestep_recent (bool, optional): If the last timestep should be the most recent data (used in RNNs)
        """
        Xtrain_temp = torch.tensor(X_neural).to(device).to(dtype)
        Ytrain_temp = torch.tensor(Y_fings).to(device).to(dtype)

        # (optional) append the previous position(s) as additional input features
        if positioninput:
            prevpos = torch.cat(
                (torch.zeros((1, numfingers)), Ytrain_temp[:-1, 0:numfingers]), dim=0
            )
            Xtrain_temp = torch.cat((Xtrain_temp, prevpos), dim=1)

        # add time delays to input features
        Xtrain1 = torch.zeros(
            (int(Xtrain_temp.shape[0]), int(Xtrain_temp.shape[1]), numdelays),
            device=device,
            dtype=dtype,
        )
        Xtrain1[:, :, 0] = Xtrain_temp
        for k1 in range(numdelays - 1):
            k = k1 + 1
            Xtrain1[k:, :, k] = Xtrain_temp[0:-k, :]

        if last_timestep_recent:
            # for RNNs, we want the last timestep to be the most recent data
            Xtrain1 = torch.flip(Xtrain1, (2,))

        # choose position/velocity/both
        if predtype == "v":
            Ytrain1 = Ytrain_temp[:, numfingers : numfingers * 2]
        if predtype == "p":
            Ytrain1 = Ytrain_temp[:, 0:numfingers]
        if predtype == "pv":
            Ytrain1 = Ytrain_temp[:, 0 : numfingers * 2]
        else:
            RuntimeError(
                "Must specify prediction type as Position 'p', Velocity 'v', or both 'pv'"
            )

        # (optional) resample velocities to based on resamp (resamp should be a vector of indices)
        if Resamp is not None and isinstance(Resamp, np.ndarray):
            # TODO - handle resampling indices (not sure if needed)
            ind1 = Resamp.astype(int)
            Xtrain = Xtrain1[ind1, :, :]
            Ytrain = Ytrain1[ind1, :]

        elif zero_train:
            ind = torch.sqrt(torch.sum(Ytrain1**2, 1)) < 0.001
            Xtrain = Xtrain1[ind, :, :]
            Ytrain = Ytrain1[ind, :]

        else:
            Xtrain = Xtrain1
            Ytrain = Ytrain1

        # store the processed X/Y data
        self.chan_states = (Xtrain, Ytrain)
        self.transform = transform

    def __len__(self):
        return len(self.chan_states[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        chans = self.chan_states[0][idx, :]
        states = self.chan_states[1][idx, :]

        sample = {"states": states, "chans": chans}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FingerDatasetMultiDay(Dataset):
    """Torch Dataset for predicting finger position/velocity from neural data.
    Same as 'FingerDataset' but also stores the day index
    """

    def __init__(
        self,
        XY_list,
        transform=None,
        zero_train=False,
        Resamp=False,
        predtype="v",
        numfingers=2,
        numdelays=3,
        positioninput=False,
        last_timestep_recent=True,
    ):
        """
        Args:
            XY_list (list): List of (X,Y) tuples, where X is Nx96 array of neural data and Y is Nx(numfings*2) array of
                            finger pos/vel data (i.e. Nx[pos1, pos2, vel1 vel2])
            batch_samps_per_run (int): number of samples per run to include in each batch
            transform (callable, optional): Optional transform to be applied on a sample.
            zero_train (bool, optional): TODO: figure out
            Resamp (bool, optional): TODO: figure out
            predtype ('v', 'p', or 'pv'): selects if position, velocity, or both are in the 'y' output
            numdelays (int, optional): Number of delay bins. Defaults to 3.
            positioninput (bool, optional): If True, the previous position is appended to the neural data as additional features
            last_timestep_recent (bool, optional): If the last timestep should be the most recent data (used in RNNs)
        """
        self.X, self.Y, self.day_idx = None, None, None
        self.num_days = len(XY_list)

        for day_num, (X_neural, Y_fings) in enumerate(XY_list):
            Xtrain_temp = torch.tensor(X_neural).to(device).to(dtype)
            Ytrain_temp = torch.tensor(Y_fings).to(device).to(dtype)

            # (optional) append the previous position(s) as additional input features
            if positioninput:
                prevpos = torch.cat(
                    (torch.zeros((1, numfingers)), Ytrain_temp[:-1, 0:numfingers]),
                    dim=0,
                )
                Xtrain_temp = torch.cat((Xtrain_temp, prevpos), dim=1)

            # add time delays to input features
            Xtrain1 = torch.zeros(
                (int(Xtrain_temp.shape[0]), int(Xtrain_temp.shape[1]), numdelays),
                device=device,
                dtype=dtype,
            )
            Xtrain1[:, :, 0] = Xtrain_temp
            for k1 in range(numdelays - 1):
                k = k1 + 1
                Xtrain1[k:, :, k] = Xtrain_temp[0:-k, :]

            if last_timestep_recent:
                # for RNNs, we want the last timestep to be the most recent data
                Xtrain1 = torch.flip(Xtrain1, (2,))

            # choose position/velocity/both
            if predtype == "v":
                Ytrain1 = Ytrain_temp[:, numfingers : numfingers * 2]
            if predtype == "p":
                Ytrain1 = Ytrain_temp[:, 0:numfingers]
            if predtype == "pv":
                Ytrain1 = Ytrain_temp[:, 0 : numfingers * 2]
            else:
                RuntimeError(
                    "Must specify prediction type as Position 'p', Velocity 'v', or both 'pv'"
                )

            # (optional) resample velocities to based on resamp (resamp should be a vector of indices)
            if Resamp is not None and isinstance(Resamp, np.ndarray):
                # TODO - handle resampling indices
                ind1 = Resamp.astype(int)
                Xtrain = Xtrain1[ind1, :, :]
                Ytrain = Ytrain1[ind1, :]

            elif zero_train:
                ind = torch.sqrt(torch.sum(Ytrain1**2, 1)) < 0.001
                Xtrain = Xtrain1[ind, :, :]
                Ytrain = Ytrain1[ind, :]

            else:
                Xtrain = Xtrain1
                Ytrain = Ytrain1

            # store the processed X/Y data along with the day number
            this_day_idx = day_num * torch.ones((Xtrain.shape[0], 1))
            this_day_idx = this_day_idx.long()
            if self.X is not None:
                self.X = torch.vstack((self.X, Xtrain.clone()))
                self.Y = torch.vstack((self.Y, Ytrain.clone()))
                self.day_idx = torch.vstack((self.day_idx, this_day_idx))
            else:
                self.X = Xtrain.clone()
                self.Y = Ytrain.clone()
                self.day_idx = this_day_idx

        # store optional transform
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "chans": self.X[idx, :],
            "states": self.Y[idx, :],
            "day_idx": self.day_idx[idx, :],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class AutoDataset(Dataset):
    """
    Sets up datasets for autoencoder (basically chans and states are the same). Very simple right now
    """

    def __init__(self, X_neural):
        """
        Args:
            X_neural (array): Nx96 array of neural data
            transform (callable, optional): Optional transform to be applied on a sample.
            zero_train (bool, optional): TODO: figure out
            Resamp (bool, optional): TODO: figure out
            last_timestep_recent (bool, optional): If the last timestep should be the most recent data (used in RNNs)
        """
        Xtrain = torch.tensor(X_neural).to(device).to(dtype)
        Ytrain = torch.tensor(X_neural).to(device).to(dtype)

        # No delays yet, time history might be best as what's done in KF for now
        self.chan_states = (Xtrain, Ytrain)

    def __len__(self):
        return len(self.chan_states[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        chans = self.chan_states[0][idx, :]
        states = self.chan_states[1][idx, :]

        sample = {"states": states, "chans": chans}

        return sample


class BasicDataset(Dataset):
    """
    Torch Dataset if your neural and behavioral data are already all set-up with history, etc. Just sets up the
    chans_states attributes and returning the sample as a dict of 'chans' and 'states'.
    """

    def __init__(self, chans, states):
        self.chans_states = (chans, states)

    def __len__(self):
        return len(self.chans_states[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        chans = self.chans_states[0][idx, :]
        states = self.chans_states[1][idx, :]

        sample = {"states": states, "chans": chans}
        return sample


def get_server_data_path():
    """
    Returns the server filepath based on what operating system is being used.
    Optional: If there's a folder called "Datasets" in the same directory as pybmi, then use that as the path. This
            makes it easier to use local datasets instead of downloading from server
    """

    # check the second parent (grandparent?) directory of pybmi for a datasets folder
    parentpath = os.path.dirname(os.path.dirname(os.path.dirname(pybmi.__file__)))
    if os.path.isdir(os.path.join(parentpath, "Datasets")):
        # use the local datasets folder
        warnings.warn("using local datasets")
        serverdatapath = os.path.join(parentpath, "Datasets")
        print(f"Datasets path = {serverdatapath}")

    else:
        # choose the standard path based on the OS
        if platform == "linux" or platform == "linux2":
            serverdatapath = "/run/user/1000/gvfs/smb-share:server=cnpl-drmanhattan.engin.umich.edu,share=share/Data/Monkeys"
        elif platform == "darwin":
            serverdatapath = "smb://cnpl-drmanhattan.engin.umich.edu/share/Data/Monkeys"
        elif platform == "win32":
            serverdatapath = "Z:/Data/Monkeys/"

    return serverdatapath


def get_cuda_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # warnings.warn('forcing cpu device')
    # device = torch.device('cpu')
    return device


device = get_cuda_device()


def get_current_time():
    # return str(datetime.datetime.now())
    return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def moving_average_normalize(data, window):
    """
    Normalizes data to 0 mean 1 std using the giving window length
    data: numpy array of size [N, numfeats]
    window: integer, number of values to normalize over
    """
    df = pd.DataFrame(data)
    df_norm = (df - df.rolling(window, min_periods=1).mean()) / df.rolling(
        window, min_periods=1
    ).std()
    data_norm = df_norm.to_numpy()
    data_norm[
        np.isnan(data_norm)
    ] = 0  # fill NaNs with 0 (i.e. the first few bins are zero)
    return data_norm


def load_training_data(
    monkey,
    date,
    run_train,
    run_test=None,
    good_chans_0idx=None,
    isrefit=False,
    fingers=[2, 4],
    binsize=32,
    batch_size=64,
    binshist=1,
    normalize_x=False,
    normalize_y=False,
    norm_x_movavg_bins=None,
    train_test_split=0.8,
    max_num_trials_train=None,
    velocity_redist=None,
    position_input=False,
    pred_type="v",
    return_norm_params=False,
    range_trials = None
):
    """Function to load in z-struct data and format for network training. Similar to getZfeats.

    Args:
        monkey (string):    'Joker'
        date (string):      '2022-01-01'
        run_train (string): 'Run-002' or ['Run-002','Run-003'] - if list then runs are concatenated
        run_test (string):  'Run-003', can be left empty and use 'train_test_split' instead
        good_chans_0idx (list): [0,1,2,3,...95]
        isrefit (bool):     True/False
        fingers (list):     [2, 4] Note: this uses matlab's 1-indexing
        binsize (int):      50
        batch_size (int):   128
        binshist (int, optional): KF should be 1. Default WillseyNet is 3.
        normalize_x (bool): if true, will normalize neural values to 0-mean 1-variance
        normalize_y (bool): if true, will normalize position/velocity values to 0-mean 1-variance
        norm_x_movavg_bins (int): number of bins to normalize neural data over using a moving average
        train_test_split (float): 0.8. if run_test is empty, will split training data based on this. 0.8 means 80% of
                                data will be for training, 20% for testing.
        max_num_trials_train (int): If not None, then this will crop the training Z struct: Z = Z(0:N)
        velocity_redist (str or None): If not None, and within current available options, will resample velocity data to
                                        most closely match a specified distirbution (tri, gauss, uni)
        position_input (bool): If True, the previous position is appended to the neural data as additional features
        pred_type ('v', 'p', or 'pv'): selects if position, velocity, or both are in the 'y' output
        return_norm_params (bool): If True, returns 1x96 arrays of Mean and Std used to normalize

    Returns:
        [loader_train, loader_val, loader_test]: pytorch laoders with the data
    """

    if good_chans_0idx is None:
        good_chans_0idx = list(range(96))

    serverdatapath = get_server_data_path()

    mask = np.zeros((96,))
    mask[good_chans_0idx] = 1

    fingers = [x - 1 for x in fingers]  # convert to 0-indexing
    finger_idx = fingers + [x + 5 for x in fingers]  # get the pos/vel for the fingers

    # get train data
    if isinstance(run_train, str):
        run_train = [run_train]

    for i, run in enumerate(run_train):
        # load zstruct for each run
        direc = os.path.join(serverdatapath, monkey, date, run)
        ztempdf = ZStructTranslator(direc, use_py=False).asdataframe()
        ztempdf = ztempdf.iloc[1:].loc[  # remove first trial (always bad)
            (ztempdf["BlankTrial"] == 0)
            & (ztempdf["DecodeFeature"] == 1)  # make sure screen on
            & (ztempdf["TrialSuccess"] == 1)  # choose SBP
        ]  # use successful trials
        zstruct_train = (
            ztempdf if i == 0 else ZTools.concatenate((zstruct_train, ztempdf))
        )

    if max_num_trials_train is not None:
        zstruct_train = zstruct_train[:max_num_trials_train]
    
    # TODO: add
    # if range_trials is not None:
    #     zstruct_train = zstruct_train[range_trials[0]:range_trials[1]]

    # get features
    feats_train = getZFeats(
        zstruct_train,
        binsize=binsize,
        featList=["FingerAnglesTIMRL", "NeuralFeature"],
        removeFirstTrial=False,
    )

    if run_test is not None:
        # load in test the run data
        direc = os.path.join(serverdatapath, monkey, date, run_test)
        zstruct_testdf = ZStructTranslator(direc, use_py=False).asdataframe()
        zstruct_testdf = zstruct_testdf.iloc[1:].loc[  # remove first trial (always bad)
            (zstruct_testdf["BlankTrial"] == 0)
            & (zstruct_testdf["DecodeFeature"] == 1)  # make sure screen on
            & (zstruct_testdf["TrialSuccess"] == 1)  # choose SBP
        ]  # use successful trials
        feats_test = getZFeats(
            zstruct_testdf,
            binsize=binsize,
            featList=["FingerAnglesTIMRL", "NeuralFeature"],
            removeFirstTrial=False,
        )
        x_train = feats_train["NeuralFeature"]
        y_train = feats_train["FingerAnglesTIMRL"]
        x_test = feats_test["NeuralFeature"]
        y_test = feats_test["FingerAnglesTIMRL"]

    else:
        # if no test run provided, split the training dataset into train/test
        num_train = int(feats_train["NeuralFeature"].shape[0] * train_test_split)
        x_train = feats_train["NeuralFeature"][:num_train, :]
        y_train = feats_train["FingerAnglesTIMRL"][:num_train, :]
        x_test = feats_train["NeuralFeature"][num_train:, :]
        y_test = feats_train["FingerAnglesTIMRL"][num_train:, :]

    # normalize X (optional)
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)  # (optional return arg)
    if normalize_x:
        if norm_x_movavg_bins:
            x_train = moving_average_normalize(x_train, norm_x_movavg_bins)
            x_test = moving_average_normalize(x_test, norm_x_movavg_bins)
            x_train, y_train = (
                x_train[norm_x_movavg_bins:, :],
                y_train[norm_x_movavg_bins:, :],
            )
            x_test, y_test = (
                x_test[norm_x_movavg_bins:, :],
                y_test[norm_x_movavg_bins:, :],
            )
        else:
            x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
            x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)

    # normalize Y (optional)
    if normalize_y:
        y_train = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)
        y_test = (y_test - y_test.mean(axis=0)) / y_test.std(axis=0)

    # mask out good chans, use only selected fingers
    x_train = mask * x_train
    y_train = y_train[:, finger_idx]
    x_test = mask * x_test
    y_test = y_test[:, finger_idx]

    # velocity redistribution - current options: 'gauss', 'tri', 'uni', None
    num_resamp = int(np.max((2e4, y_train.shape[0])))
    vy = y_train[:, int(y_train.shape[1] / 2) :]  # pull off velocities
    ny = (vy - np.mean(vy)) / np.std(vy)
    # all specific values here were in willsey's old code.
    if isinstance(velocity_redist, str):
        if velocity_redist == "gauss":
            pd_vec = 2 * np.random.randn((num_resamp, len(fingers)))
        elif velocity_redist == "tri":
            pd_vec = np.random.triangular(-4, 0, 4, (num_resamp, len(fingers)))
        elif velocity_redist == "uni":
            pd_vec = np.random.uniform(-4, 4, (num_resamp, len(fingers)))
        else:
            Warning("Not a valid distribution defaulting to None")
            pd_vec = None
    elif velocity_redist == None:
        pd_vec = None
    else:
        Warning("Not a valid distribution, defaulting to None")
        pd_vec = None

    if pd_vec is not None:
        kdt = KDTree(ny)
        idx = kdt.query(pd_vec)[1]
    else:
        idx = None

    # report back
    print(f"loaded {x_train.shape[0]} training samples")
    print(f"loaded {x_test.shape[0]} validation samples")
    if idx is not None:
        print(
            f"generated {len(idx)} resampling indices for distribution {velocity_redist}"
        )

    # setup datasets (which add time history)
    dataset_train = FingerDataset(
        X_neural=x_train,
        Y_fings=y_train,
        predtype=pred_type,
        numfingers=len(fingers),
        numdelays=binshist,
        positioninput=position_input,
        last_timestep_recent=True,
        Resamp=idx,
    )
    dataset_test = FingerDataset(
        X_neural=x_test,
        Y_fings=y_test,
        predtype=pred_type,
        numfingers=len(fingers),
        numdelays=binshist,
        positioninput=position_input,
        last_timestep_recent=True,
    )
    # setup dataloaders
    num_train = len(dataset_train)
    num_test = len(dataset_test)
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=sampler.RandomSampler(range(num_train)),
        drop_last=True,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=num_test,
        sampler=sampler.SequentialSampler(range(num_test)),
    )

    if return_norm_params:
        return loader_train, loader_test, x_mean, x_std
    else:
        return loader_train, loader_test


def load_training_data_refit(
    monkey,
    date,
    run_train,
    good_chans_0idx=None,
    fingers=[2, 4],
    binsize=32,
    batch_size=64,
    binshist=1,
    normalize_x=False,
    norm_x_movavg_bins=None,
):
    """Function to load in z-struct data and format for REFIT network training. Very similar to `load_training_data`.

    Args:
        monkey (string):    'Joker'
        date (string):      '2022-01-01'
        run_train (string): 'Run-002' or ['Run-002','Run-003'] - if list then runs are concatenated
        good_chans_0idx (list): [0,1,2,3,...95]
        fingers (list):     [2, 4] Note: this uses matlab's 1-indexing
        binsize (int):      50
        batch_size (int):   128
        binshist (int, optional): KF should be 1. Default WillseyNet is 3.
        normalize_x (bool): if true, will normalize neural values to 0-mean 1-variance
        norm_x_movavg_bins (int): number of bins to normalize neural data over using a moving average

    Returns:
        [loader_train, loader_val, loader_test]: pytorch laoders with the data
    """
    if good_chans_0idx is None:
        good_chans_0idx = list(range(96))

    serverdatapath = get_server_data_path()

    mask = np.zeros((96,))
    mask[good_chans_0idx] = 1

    fingers = [x - 1 for x in fingers]  # convert to 0-indexing
    finger_idx = fingers + [x + 5 for x in fingers]  # get the pos/vel for the fingers

    # get train data
    direc = os.path.join(serverdatapath, monkey, date, run_train)
    z = ZStructTranslator(direc, use_py=False)
    zStructDF = z.asdataframe()
    zStructDF = zStructDF.iloc[1:]  # remove first trial (always bad)
    zStructDF = zStructDF.loc[zStructDF["BlankTrial"] == 0]  # make sure screen on
    zStructDF = zStructDF.loc[zStructDF["ClosedLoop"] == 1]  # use online trials
    zStructDF = zStructDF.loc[zStructDF["DecodeFeature"] == 1]  # choose SBP
    zStructDF = zStructDF.loc[zStructDF["TrialSuccess"] == 1]  # use successful trials

    # get features
    featList = [
        "FingerAnglesTIMRL",
        "NeuralFeature",
        "Decode",
        "TargetPos",
        "TargetScaling",
    ]
    feats_train = getZFeats(
        zStructDF,
        binsize=binsize,
        featList=featList,
        trimBlankTrials=False,
        removeFirstTrial=False,
    )

    # setup velocities and target positions
    targ_pos = feats_train["TargetPos"]
    targ_scales = feats_train["TargetScaling"]
    targ_sizes = 0.0375 * (1 + targ_scales / 100)
    ypos = feats_train["Decode"][
        :, 5:
    ]  # get decoded SBP positions (0-4 are TCFR, 5-9 are SBP)
    tempvelocity = np.diff(ypos, n=1, axis=0)
    yvel = np.concatenate(
        (tempvelocity, np.zeros([1, ypos.shape[1]])), axis=0
    )  # the baseline decoded velocities (Nx5)

    # ----------------- REFIT ---------------------------
    # zero velocities in target
    yvel_flip = yvel
    in_target_idx = np.abs(ypos - targ_pos) < targ_sizes
    yvel_flip[in_target_idx] = 0

    # flip incorrect velocities
    dist2targ = targ_pos - ypos
    yvel_flip = np.sign(dist2targ) * np.sign(yvel_flip) * yvel_flip
    # ---------------------------------------------------

    # format train matrices (test data is the same as train data for refit since we don't have another set)
    x_train = feats_train["NeuralFeature"]
    y_train = np.hstack(
        (np.zeros_like(yvel_flip), yvel_flip)
    )  # append dummy position data

    # normalize X (optional)
    if normalize_x:
        if norm_x_movavg_bins:
            x_train = moving_average_normalize(x_train, norm_x_movavg_bins)
            x_train, y_train = (
                x_train[norm_x_movavg_bins:, :],
                y_train[norm_x_movavg_bins:, :],
            )
        else:
            x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)

    # mask out good chans, use only selected fingers
    x_train = mask * x_train
    y_train = y_train[:, finger_idx]
    x_test = x_train
    y_test = y_train

    # setup datasets (which add time history)
    dataset_train = FingerDataset(
        X_neural=x_train,
        Y_fings=y_train,
        predtype="v",
        numfingers=len(fingers),
        numdelays=binshist,
        last_timestep_recent=True,
    )
    dataset_test = FingerDataset(
        X_neural=x_test,
        Y_fings=y_test,
        predtype="v",
        numfingers=len(fingers),
        numdelays=binshist,
        last_timestep_recent=True,
    )
    # setup dataloaders
    num_train = len(dataset_train)
    num_test = len(dataset_test)
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=sampler.RandomSampler(range(num_train)),
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=num_test,
        sampler=sampler.SequentialSampler(range(num_test)),
    )

    return [loader_train, loader_test]


def load_training_data_multiday(
    monkey,
    day_run_list,
    good_chans_0idx=None,
    fingers=[2, 4],
    binsize=32,
    batch_size=64,
    binshist=1,
    normalize_x=False,
    normalize_y=False,
    norm_x_movavg_bins=None,
    train_test_split=0.8,
    max_num_trials_train=None,
    velocity_redist=None,
    position_input=False,
    pred_type="v",
):
    """Function to load in z-struct data and format for network training. Similar to getZfeats. This version (multiday)
        loads in multiple days of data and returns a multi-day dataloader.

    Args:
        monkey (string):    'Joker'
        day_run_list (list): List of tuples for each day (date, run-train, run-test). ex: ('2022-04-01', 'Run-003', None)
                            If run-test is None, then the training run will be split by 'train_test_split'
        good_chans_0idx (list): [0,1,2,3,...95]
        isrefit (bool):     True/False
        fingers (list):     [2, 4] Note: this uses matlab's 1-indexing
        binsize (int):      50
        batch_size (int):   128
        binshist (int, optional): KF should be 1. Default WillseyNet is 3.
        normalize_x (bool): if true, will normalize neural values to 0-mean 1-variance
        normalize_y (bool): if true, will normalize position/velocity values to 0-mean 1-variance
        norm_x_movavg_bins (int): number of bins to normalize neural data over using a moving average
        train_test_split (float): 0.8. if run_test is empty, will split training data based on this. 0.8 means 80% of
                                data will be for training, 20% for testing.
        max_num_trials_train (int): If not None, then this will crop the training Z struct: Z = Z(0:N)
        velocity_redist (str or None): If not None, and within current available options, will resample velocity data to
                                        most closely match a specified distirbution (tri, gauss, uni)
        position_input (bool): If True, the previous position is appended to the neural data as additional features
        pred_type ('v', 'p', or 'pv'): selects if position, velocity, or both are in the 'y' output

    Returns:
        [loader_train, loader_val, loader_test]: pytorch laoders with the data
    """

    if good_chans_0idx is None:
        good_chans_0idx = list(range(96))

    serverdatapath = get_server_data_path()

    mask = np.zeros((96,))
    mask[good_chans_0idx] = 1

    fingers = [x - 1 for x in fingers]  # convert to 0-indexing
    finger_idx = fingers + [x + 5 for x in fingers]  # get the pos/vel for the fingers

    # init lists to hold data (filled in below)
    XY_list_train = []
    XY_list_test = []

    # loop over each day's data
    for date, run_train, run_test in day_run_list:

        # load training data
        direc = os.path.join(serverdatapath, monkey, date, run_train)
        zstruct_train = ZStructTranslator(direc, use_py=False).asdataframe()
        zstruct_train = zstruct_train.iloc[1:].loc[  # remove first trial (always bad)
            (zstruct_train["BlankTrial"] == 0)
            & (zstruct_train["DecodeFeature"] == 1)  # make sure screen on
            & (zstruct_train["TrialSuccess"] == 1)  # choose SBP
        ]  # use successful trials

        if max_num_trials_train is not None:
            zstruct_train = zstruct_train[:max_num_trials_train]

        # get features
        feats_train = getZFeats(
            zstruct_train,
            binsize=binsize,
            featList=["FingerAnglesTIMRL", "NeuralFeature"],
            removeFirstTrial=False,
        )

        if run_test is not None:
            # load in test data
            direc = os.path.join(serverdatapath, monkey, date, run_test)
            zstruct_testdf = ZStructTranslator(direc, use_py=False).asdataframe()
            zstruct_testdf = zstruct_testdf.iloc[
                1:
            ].loc[  # remove first trial (always bad)
                (zstruct_testdf["BlankTrial"] == 0)
                & (zstruct_testdf["DecodeFeature"] == 1)  # make sure screen on
                & (zstruct_testdf["TrialSuccess"] == 1)  # choose SBP
            ]  # use successful trials
            feats_test = getZFeats(
                zstruct_testdf,
                binsize=binsize,
                featList=["FingerAnglesTIMRL", "NeuralFeature"],
                removeFirstTrial=False,
            )
            x_train = feats_train["NeuralFeature"]
            y_train = feats_train["FingerAnglesTIMRL"]
            x_test = feats_test["NeuralFeature"]
            y_test = feats_test["FingerAnglesTIMRL"]

        else:
            # if no test run provided, split the training dataset into train/test
            num_train = int(feats_train["NeuralFeature"].shape[0] * train_test_split)
            x_train = feats_train["NeuralFeature"][:num_train, :]
            y_train = feats_train["FingerAnglesTIMRL"][:num_train, :]
            x_test = feats_train["NeuralFeature"][num_train:, :]
            y_test = feats_train["FingerAnglesTIMRL"][num_train:, :]

        # normalize X (optional)
        if normalize_x:
            if norm_x_movavg_bins:
                x_train = moving_average_normalize(x_train, norm_x_movavg_bins)
                x_test = moving_average_normalize(x_test, norm_x_movavg_bins)
                x_train, y_train = (
                    x_train[norm_x_movavg_bins:, :],
                    y_train[norm_x_movavg_bins:, :],
                )
                x_test, y_test = (
                    x_test[norm_x_movavg_bins:, :],
                    y_test[norm_x_movavg_bins:, :],
                )
            else:
                x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
                x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)

        # normalize Y (optional)
        if normalize_y:
            y_train = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)
            y_test = (y_test - y_test.mean(axis=0)) / y_test.std(axis=0)

        # mask out good chans, use only selected fingers
        x_train = mask * x_train
        y_train = y_train[:, finger_idx]
        x_test = mask * x_test
        y_test = y_test[:, finger_idx]

        # velocity redistribution - current options: 'gauss', 'tri', 'uni', None
        num_resamp = int(np.max((2e4, y_train.shape[0])))
        vy = y_train[:, int(y_train.shape[1] / 2) :]  # pull off velocities
        ny = (vy - np.mean(vy)) / np.std(vy)
        # all specific values here were in willsey's old code.
        if isinstance(velocity_redist, str):
            if velocity_redist == "gauss":
                pd_vec = 2 * np.random.randn((num_resamp, len(fingers)))
            elif velocity_redist == "tri":
                pd_vec = np.random.triangular(-4, 0, 4, (num_resamp, len(fingers)))
            elif velocity_redist == "uni":
                pd_vec = np.random.uniform(-4, 4, (num_resamp, len(fingers)))
            else:
                Warning("Not a valid distribution defaulting to None")
                pd_vec = None
        elif velocity_redist == None:
            pd_vec = None
        else:
            Warning("Not a valid distribution, defaulting to None")
            pd_vec = None

        if pd_vec is not None:
            kdt = KDTree(ny)
            idx = kdt.query(pd_vec)[1]
        else:
            idx = None

        # report back
        # print(f'{date}: loaded {x_train.shape[0]} training samples')
        # print(f'{date}: loaded {x_test.shape[0]} validation samples')
        if idx is not None:
            print(
                f"generated {len(idx)} resampling indices for distribution {velocity_redist}"
            )

        # save data to list
        XY_list_train.append((x_train, y_train))
        XY_list_test.append((x_test, y_test))

    # setup datasets (which add time history)
    dataset_train = FingerDatasetMultiDay(
        XY_list=XY_list_train,
        predtype=pred_type,
        numfingers=len(fingers),
        numdelays=binshist,
        positioninput=position_input,
        last_timestep_recent=True,
        Resamp=idx,
    )
    dataset_test = FingerDatasetMultiDay(
        XY_list=XY_list_test,
        predtype=pred_type,
        numfingers=len(fingers),
        numdelays=binshist,
        positioninput=position_input,
        last_timestep_recent=True,
        Resamp=idx,
    )

    print(f"loaded {len(dataset_train)} training samples")
    print(f"loaded {len(dataset_test)} test samples")

    # setup dataloaders
    num_train = len(dataset_train)
    num_test = len(dataset_test)
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=sampler.RandomSampler(range(num_train)),
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=num_test,
        sampler=sampler.SequentialSampler(range(num_test)),
    )

    return loader_train, loader_test


def load_training_data_auto(
    monkey,
    date,
    run_train,
    run_test=None,
    good_chans_0idx=None,
    isrefit=False,
    binsize=32,
    batch_size=64,
    binshist=1,
    normalize_x=False,
    train_test_split=0.8,
    max_num_trials_train=None,
    velocity_redist=None,
):
    if good_chans_0idx is None:
        good_chans_0idx = list(range(96))

    serverdatapath = get_server_data_path()

    mask = np.zeros((96,))
    mask[good_chans_0idx] = 1

    # get train data
    if isinstance(run_train, str):
        run_train = [run_train]

    for i, run in enumerate(run_train):
        # load zstruct for each run
        direc = os.path.join(serverdatapath, monkey, date, run)
        ztemp = ZStructTranslator(direc, use_py=False)
        # mask out successful trials
        sucMask = np.zeros(len(ztemp), dtype=bool)
        for j, trial in enumerate(ztemp):
            sucMask[j] = trial.TrialSuccess
        if i == 0:
            zstruct_train = ztemp[sucMask]
        else:
            # append to current max trials
            zstruct_train = ZTools.concatenate((zstruct_train, ztemp[sucMask]))

    if max_num_trials_train is not None:
        zstruct_train = zstruct_train[:max_num_trials_train]

    # get features
    feats_train = getZFeats(zstruct_train, binsize=binsize, featList=["NeuralFeature"])

    if run_test is not None:
        # load in test the run data
        direc = os.path.join(serverdatapath, monkey, date, run_test)
        zstruct_test = ZStructTranslator(direc, use_py=False)
        # mask out successful trials
        sucMask = np.zeros(len(zstruct_test), dtype=bool)
        for i, trial in enumerate(zstruct_test):
            sucMask[i] = trial.TrialSuccess
        zstruct_test = zstruct_test[sucMask]

        feats_test = getZFeats(
            zstruct_test, binsize=binsize, featList=["NeuralFeature"]
        )

        x_train = feats_train["NeuralFeature"]
        y_train = feats_train["NeuralFeature"]
        x_test = feats_test["NeuralFeature"]
        y_test = feats_test["NeuralFeature"]

    else:  # if no test run provided
        # split the training dataset into train/test
        num_train = int(feats_train["NeuralFeature"].shape[0] * train_test_split)
        x_train = feats_train["NeuralFeature"][:num_train, :]
        y_train = feats_train["NeuralFeature"][:num_train, :]
        x_test = feats_train["NeuralFeature"][num_train:, :]
        y_test = feats_train["NeuralFeature"][num_train:, :]

    # normalize X (optional)
    if normalize_x:
        x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)

    # mask out good chans, use only selected fingers
    x_train = mask * x_train
    y_train = mask * y_train
    x_test = mask * x_test
    y_test = mask * y_test

    # report back
    print(f"loaded {x_train.shape[0]} training samples")
    print(f"loaded {x_test.shape[0]} validation samples")

    # setup datasets (which add time history)
    dataset_train = AutoDataset(X_neural=x_train)
    dataset_test = AutoDataset(X_neural=x_test)
    # setup dataloaders
    num_train = len(dataset_train)
    num_test = len(dataset_test)
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=sampler.RandomSampler(range(num_train)),
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=num_test,
        sampler=sampler.SequentialSampler(range(num_test)),
    )
    return [loader_train, loader_test]


# TODO replace with a better function or Tools version
def calc_corr(y1, y2):
    """Calculates the correlation between y1 and y2 (tensors)"""
    corr = []
    y1 = y1.cpu()
    y2 = y2.cpu()
    for i in range(y1.shape[1]):
        corr.append(np.corrcoef(y1[:, i], y2[:, i])[1, 0])
    return corr


def run_model_forward(model, loader, multiday=False):
    """Runs the model using the provided dataloader"""
    with torch.no_grad():
        # get batch data (we assume there's only 1 batch) TODO: concat multiple batches
        for batch in loader:
            x = batch["chans"].to(device=device)
            y = batch["states"].to(device=device)
            model.eval()

            if multiday:
                day_idx = batch["day_idx"].to(device=device)
                yhat = model.forward(x, day_idx)  # forward pass for multiday training
            else:
                yhat = model.forward(x)  # normal forward pass

            if isinstance(yhat, tuple):
                # RNNs return y, h
                yhat = yhat[0]
            return y, yhat


def check_accuracy(model, loader, loss_func, normalize_y=False, multiday=False):
    """Calculates the loss and correlation given a model and dataset."""
    y, yhat = run_model_forward(model, loader, multiday=multiday)
    loss = loss_func(yhat, y).item()
    if normalize_y:
        y = (y - y.mean(dim=0)) / y.var(dim=0)
        yhat = (yhat - yhat.mean(dim=0)) / yhat.var(dim=0)
    corr = calc_corr(y, yhat)
    return loss, corr


def plot_train_progress(
    model,
    loader,
    loss_history_train,
    loss_history_val,
    corr_history_val,
    valloss,
    corr,
    plot_block=False,
    autoscaleyhat=True,
    tlim=20,
    binsize=50,
):
    """Plots training stats while training is happening and updates the current plot. Some finagling is needed to have
        the plot stay open when other tasks continue.

    Args:
        model (decoder):            a decoder
        loader (Dataloader):        data to plot
        loss_history_train (list):  list of training loss values for each iter
        loss_history_val (list):    list of validation loss values for each iter
        corr_history_val (list):    list of validation correlations for each iter
        valloss (float):            most recent validation loss
        corr (float):               most recent correlation
        plot_block (bool, optional: if the plot should block (pause) all further execution
        autoscaleyhat (bool, optional): Autoscale the predicted values for same magnitude as true values
        tlim (int, optional):       Max time to plot. Defaults to 20.
        binsize (int, optional):    Defaults to 50.
    """

    if plot_train_progress.is_first_plot:
        # https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
        plot_train_progress.fig, plot_train_progress.ax = plt.subplots(
            nrows=3, ncols=4, figsize=(14, 6)
        )

        # setup the yhat plot (bottom 2 rows)
        gs = plot_train_progress.ax[0, 0].get_gridspec()
        for row in plot_train_progress.ax[1:, :]:
            for ax in row:
                ax.remove()
        plot_train_progress.axyhat1 = plot_train_progress.fig.add_subplot(gs[1, :])
        plot_train_progress.axyhat2 = plot_train_progress.fig.add_subplot(gs[2, :])

        plt.ion()
        plt.show()
        plot_train_progress.is_first_plot = False

    colors = ("#1f77b4", "#ff7f0e")

    # MSE plot - train
    plot_train_progress.ax[0, 0].lines = []
    plot_train_progress.ax[0, 0].plot(loss_history_train, color=[0.1, 0.1, 0.1])
    plot_train_progress.ax[0, 0].set_title("Train Loss")
    plot_train_progress.ax[0, 0].set_ylabel("MSE")
    plot_train_progress.ax[0, 0].set_xlabel("Iter")

    # MSE plot - validation
    plot_train_progress.ax[0, 1].lines = []
    plot_train_progress.ax[0, 1].plot(loss_history_val, color=[0.1, 0.1, 0.1])
    plot_train_progress.ax[0, 1].set_title("Val Loss - Normalized Y")
    plot_train_progress.ax[0, 1].set_ylabel("MSE")
    plot_train_progress.ax[0, 1].set_xlabel("Iter")

    # Corr plot
    plot_train_progress.ax[0, 2].lines = []
    plot_train_progress.ax[0, 2].set_prop_cycle(None)  # reset color order
    plot_train_progress.ax[0, 2].plot(corr_history_val)
    plot_train_progress.ax[0, 2].set_title(
        str([round(x, 3) for x in corr_history_val[-1]])
    )
    plot_train_progress.ax[0, 2].set_ylabel("Correlation")
    plot_train_progress.ax[0, 2].set_xlabel("Iter")

    # Velocity errors plot
    plot_train_progress.ax[0, 3].lines = []
    y, yhat = run_model_forward(model, loader)
    y = y.cpu().detach().numpy()
    yhat = yhat.cpu().detach().numpy()
    Nbins = 21
    bins = np.linspace(-0.2, 0.2, Nbins)
    for i in range(y.shape[1]):
        ybinInds = np.digitize(y[:, i], bins)  # put in bins
        MAE = np.zeros(bins.shape)
        for binnum, bin in enumerate(bins):
            idx = ybinInds == binnum  # get yvals in this bin
            MAE[binnum] = np.mean(
                np.abs(yhat[idx, i] - y[idx, i])
            )  # get avg error for bin
            binscentered = bins + (bins[1] - bins[0]) / 2
        plot_train_progress.ax[0, 3].plot(
            binscentered, np.divide(MAE, np.abs(binscentered)), color=colors[i]
        )
    plot_train_progress.ax[0, 3].set_xlabel("True Velocity")
    plot_train_progress.ax[0, 3].set_ylabel("MAE / Vel")

    # Plot true vs predicted velocities
    t = np.arange(y.shape[0]) * binsize / 1000
    if autoscaleyhat:
        for i in range(yhat.shape[1]):
            gain = np.std(y[:, i]) / np.std(yhat[:, i] - np.median(yhat[:, i]))
            yhat[:, i] = (yhat[:, i] - np.median(yhat[:, i])) * gain
    plot_train_progress.axyhat1.lines = []
    plot_train_progress.axyhat1.plot(t, y[:, 0], color=colors[0])
    plot_train_progress.axyhat1.plot(t, yhat[:, 0], color=colors[1])
    plot_train_progress.axyhat1.set_xlim((0, tlim))
    plot_train_progress.axyhat1.set_ylim((1.2 * np.max(y[:, 0]), 1.2 * np.min(y[:, 0])))
    plot_train_progress.axyhat1.legend(["True Vel", "NN Vel"], loc="upper right")
    plot_train_progress.axyhat1.set_title(f"autoscaled = {autoscaleyhat}")
    if y.shape[1] > 1:
        plot_train_progress.axyhat2.lines = []
        plot_train_progress.axyhat2.plot(t, y[:, 1], color=colors[0])
        plot_train_progress.axyhat2.plot(t, yhat[:, 1], color=colors[1])
        plot_train_progress.axyhat2.set_xlim((0, tlim))
        plot_train_progress.axyhat2.set_ylim(
            (1.2 * np.max(y[:, 1]), 1.2 * np.min(y[:, 1]))
        )

    # Update the plot
    plot_train_progress.fig.tight_layout()
    if not plot_block:
        plt.draw()
        plt.pause(0.001)
    else:
        plt.show(block=True)
    print("Test Correlation:" + str(corr))
    print("MSE:" + str(valloss))


def plot_train_progress_simple(
    loss_history_train, loss_history_val, corr_history_val, val_loss, corr
):
    """Function to make a basic plot showing training/validation loss across iterations
    Args:
        model (decoder):            a decoder
        loader (Dataloader):        data to plot
        loss_history_train (list):  list of training loss values for each iter
        loss_history_val (list):    list of validation loss values for each iter
        corr_history_val (list):    list of validation correlations for each iter
        valloss (float):            most recent validation loss
        corr (float):               most recent correlation
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(loss_history_train)
    ax1.plot(loss_history_val)
    ax1.set_title("Loss (train, val)")
    ax1.set_ylabel("MSE")
    ax1.set_xlabel("Iter")
    ax1.set_yscale("log")

    # corrsIdx, corrsMRP = zip(*corr_history_val)
    # ax2.plot(corrsIdx)
    # ax2.plot(corrsMRP)
    ax2.plot(corr_history_val)
    ax2.set_title("Correlation (idx, mrp)")
    ax2.set_ylabel("Corr")
    ax2.set_xlabel("Iter")

    plt.show()

    print("Test Correlation:" + str(corr))
    print("MSE:" + str(val_loss))


def add_training_noise(
    x,
    bias_neural_std=None,
    noise_neural_std=None,
    noise_neural_walk_std=None,
    bias_allchans_neural_std=None,
    device="cpu",
):
    """Function to add different types of noise to training input data to make models more robust.
       Identical to the methods in Willet 2021.
    Args:
        x (tensor):                     neural data of shape [batch_size x num_chans x conv_size]
        bias_neural_std (float):        std of bias noise
        noise_neural_std (float):       std of white noise
        noise_neural_walk_std (float):  std of random walk noise
        bias_allchans_neural_std (float): std of bias noise, bias is same across all channels
        device (device):                torch device (cpu or cuda)
    """
    if bias_neural_std:
        # bias is constant across time (i.e. the 3 conv inputs), but different for each channel & batch
        # biases = torch.normal(0, bias_neural_std, x.shape[:2]).unsqueeze(2).repeat(1, 1, x.shape[2])
        biases = (
            torch.normal(torch.zeros(x.shape[:2]), bias_neural_std)
            .unsqueeze(2)
            .repeat(1, 1, x.shape[2])
        )
        x = x + biases.to(device=device)

    if noise_neural_std:
        # adds white noise to each channel and timepoint (independent)
        # noise = torch.normal(0, noise_neural_std, x.shape)
        noise = torch.normal(torch.zeros_like(x), noise_neural_std)
        x = x + noise.to(device=device)

    if noise_neural_walk_std:
        # adds a random walk to each channel (noise is summed across time)
        # noise = torch.normal(0, noise_neural_walk_std, x.shape).cumsum(dim=2)
        noise = torch.normal(torch.zeros_like(x), noise_neural_walk_std).cumsum(dim=2)
        x = x + noise.to(device=device)

    if bias_allchans_neural_std:
        # bias is constant across time (i.e. the 3 conv inputs), and same for each channel
        biases = torch.normal(
            torch.zeros((x.shape[0], 1, 1)), bias_allchans_neural_std
        ).repeat(1, x.shape[1], x.shape[2])
        x = x + biases.to(device=device)

    return x


def train_model(
    model,
    optimizer,
    loss_func,
    loader_train,
    loader_val,
    check_accuracy_iters=100,
    verbose=True,
    scheduler=None,
    min_lr=None,
    max_iter=None,
    max_epoch=500,
    plot_progress=True,
    plot_simple=False,
    sparsify_model=False,
    sparse_pctiles=[],
    sparse_iters=[],
    return_history=False,
    bias_neural_std=None,
    noise_neural_std=None,
    noise_neural_walk_std=None,
    multiday=False,
):
    """Trains any neural network model using standard gradient descent.
        Works with FNNs and RNNs, can sparsify the network, can add noise to train data, and can use multiday-dataloaders.
        If plot_progress is true, will plot train and validation loss as training progresses.

    Args:
        model:                          pytorch model
        optimizer:                      pytorch optimizer (Adam)
        loss_func:                      loss func (nn.mseloss)
        loader_train ([Dataloader):     training laoder
        loader_val (Dataloader):        validation loader
        check_accuracy_iters (int):     [description]. Defaults to 100
        verbose (bool, optional):       if should print training updates
        scheduler (optional):           Learning rate scheduler. If provided, won't use the 'max_iter'
        min_lr (float, optional):       Training will stop when the scheduler reaches this learning rate
        max_iter (int, optional):       Iteration to stop training. Used if no scheduler provided
        max_epoch (int, optional):      Max epoch. Will stop after this epoch.
        plot_progress (bool, optional): If should plot progress during training
        plot_simple (bool, optional):   True - use live in-depth plot. False - simple mse/corr plot.
        sparsify_model (bool, optional): If the model should be sparsified
        sparse_pctiles (list, optional): If sparsifying, what percentiles to use at each step. [60, 70, 80, 90, 95]
        sparse_iters (list, optional):  If sparsifying, which iter to sparsify at [2500, 2700, 2900, 3100, 3300]
        return_history (bool, optional): If true, will return training history (train loss, val loss, val corr) as additional outputs.
        bias_neural_std (float, optional):       Adds a random bias to neural data to improve robustness
        noise_neural_std (float, optional):      Adds a white noise to neural data to improve robustness
        noise_neural_walk_std (float, optional): Adds random walk white noise to neural data to improve robustness
        multiday (bool, optional):      If true, uses multiday info. Dataloaders and models must support it.

    Returns:
        [model, iter, valloss, corr]:   trained model, final iteration, validation loss, validation correlation, (train loss history, val loss history, val corr history) - optional
    """

    plotCnt = 0
    valloss = 0
    corr = 0
    loss_history_train = []
    loss_history_val = []
    corr_history_val = []
    plot_train_progress.is_first_plot = True
    iter = 0
    sparse_step = 0

    for epoch in range(max_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        i = 0

        for batch in loader_train:
            x = batch["chans"]  # [batch_size x 96 x conv_size]
            y = batch["states"]  # [batch_size x num_fings]
            x = x.to(device=device)
            y = y.to(device=device)
            model.train()

            # (optional) add noise to improve training robustness
            if bias_neural_std or noise_neural_std or noise_neural_walk_std:
                x = add_training_noise(
                    x,
                    bias_neural_std,
                    noise_neural_std,
                    noise_neural_walk_std,
                    None,
                    device,
                )

            # zero gradients + forward + backward + optimize
            optimizer.zero_grad()
            if multiday:
                day_idx = batch["day_idx"].to(device=device)
                yhat = model.forward(x, day_idx)  # forward pass for multiday training
            else:
                yhat = model.forward(x)  # normal forward pass

            if isinstance(yhat, tuple):
                # RNNs return y, h
                yhat = yhat[0]

            loss = loss_func(yhat, y)
            loss.backward()
            if sparsify_model:
                model.zeroGradients()
            optimizer.step()

            # keep track of iteration and loss
            i += 1
            iter += 1
            running_loss += loss.item()
            plotCnt += 1

            # sparsify the model if desired
            if sparsify_model:
                if iter >= sparse_iters[sparse_step]:
                    print(
                        f"Thresholded step={sparse_step}, iter={iter}, pctile={sparse_pctiles[sparse_step]}"
                    ) if verbose else None
                    model.thresholdWeights(sparse_pctiles[sparse_step])
                    sparse_step += 1

            # occasionally check validation accuracy and plot
            if plotCnt >= check_accuracy_iters:
                plotCnt = 0
                valloss, corr = check_accuracy(
                    model, loader_val, loss_func, normalize_y=True, multiday=multiday
                )
                loss_history_val.append(valloss)
                corr_history_val.append(corr)
                loss_history_train.append(running_loss / i)

                if verbose:
                    print(f"iter {iter}: val corr = {corr}")

                # plot
                if plot_progress:
                    if plot_simple:
                        plot_train_progress_simple(
                            loss_history_train,
                            loss_history_val,
                            corr_history_val,
                            valloss,
                            corr,
                        )
                    else:
                        plot_train_progress(
                            model,
                            loader_val,
                            loss_history_train,
                            loss_history_val,
                            corr_history_val,
                            valloss,
                            corr,
                            False,
                            autoscaleyhat=True,
                            tlim=20,
                            binsize=32,
                        )

                # if there's an lr scheduler then use it
                if scheduler is not None:
                    scheduler.step(valloss)
                    if optimizer.param_groups[0]["lr"] < float(min_lr):
                        print(
                            "*** model done improving based on scheduler ***"
                        ) if verbose else None
                        if return_history:
                            return (
                                model,
                                iter,
                                valloss,
                                corr,
                                (
                                    loss_history_train,
                                    loss_history_val,
                                    corr_history_val,
                                ),
                            )
                        else:
                            return model, iter, valloss, corr

                # if there's a max iteration then stop if we've reached it
                elif max_iter is not None:
                    if iter >= max_iter:
                        print(
                            f"*** model done - stopped at iteration {iter} ***"
                        ) if verbose else None
                        if return_history:
                            return (
                                model,
                                iter,
                                valloss,
                                corr,
                                (
                                    loss_history_train,
                                    loss_history_val,
                                    corr_history_val,
                                ),
                            )
                        else:
                            return model, iter, valloss, corr

        # loss_history_train.append(running_loss / i)
    print("*** final epoch is done ***") if verbose else None
    if return_history:
        return (
            model,
            iter,
            valloss,
            corr,
            (loss_history_train, loss_history_val, corr_history_val),
        )
    else:
        return model, iter, valloss, corr


def calc_gain_peaks(model, loader):
    """
    Calculates the peak/scale ratios and medians for signals. Treats each dimension independently. Adapted from willsey's
    code. Does not include SNR calculations, only finding peaks/scale ratios.
    """
    model.eval()  # set model to evaluation mode
    batches = len(list(loader))
    with torch.no_grad():
        for k1 in range(batches):
            temp = list(loader)
            x = temp[k1]["chans"].to(
                device=device, dtype=dtype
            )  # move to device, e.g. GPU
            y = temp[k1]["states"].to(device=device, dtype=dtype)

            scores = model(x).cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            d = scores.shape[1]
            peaks_hat = np.zeros(scores.shape)
            peaks_sig = np.zeros(y.shape)
            avg_peak_yhat = np.zeros((d, 1))
            avg_peak_ysig = np.zeros((d, 1))

            peakratio = np.zeros((d, 1))
            medians = np.median(scores, axis=0)
            # for each dimension, calculate the peaks of the signal
            for dim in range(d):
                np_temp = 1.5 * np.median(np.abs(scores[:, dim]))
                peaks_ind, properties = find_peaks(
                    np.abs(scores[:, dim]), height=3 * np_temp
                )
                peaks_hat[peaks_ind, dim] = 1
                avg_peak_yhat[dim] = np.mean(np.abs(scores[peaks_ind, dim]))

                np_temp = 1.5 * np.median(np.abs(y[:, dim]))
                peaks_ind_sig, properties = find_peaks(
                    np.abs(y[:, dim]), height=3 * np_temp
                )
                peaks_sig[peaks_ind_sig, dim] = 1
                avg_peak_ysig[dim] = np.mean(np.abs(y[peaks_ind_sig, dim]))

                peakratio[dim] = avg_peak_ysig[dim] / avg_peak_yhat[dim]
    return peakratio, medians


# TODO - ADD RIDGE REGRESSION/K-FOLD CV
def calcGainRR(loader, model, idpt=True, subtract_median=False, verbose=True):
    model.eval()  # set model to evaluation mode
    batches = len(list(loader))
    with torch.no_grad():
        for k1 in range(batches):  # TODO: handle multiple batches in loader
            temp = list(loader)
            x = temp[k1]["chans"]
            # x = x[:, :, 0:ConvSize]
            y = temp[k1]["states"]
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            yhat = model(x)
            if isinstance(yhat, tuple):
                # RNNs return y, h
                yhat = yhat[0]
            # if ConvSize == 1:
            #     scores = model(x[:, :, 0:ConvSize])
            # else:
            #     scores = model(x[:, :, 0:ConvSize])

            medians = []
            if subtract_median:
                for i in range(yhat.shape[1]):
                    medians.append(torch.median(yhat[:, i]))
                    yhat[:, i] = yhat[:, i] - torch.median(yhat[:, i])

            num_samps = yhat.shape[0]
            num_outputs = yhat.shape[1]
            yh_temp = torch.cat((yhat, torch.ones([num_samps, 1]).to(device)), dim=1)

            # # TEMP:
            # print('yh_temp shape:')
            # print(yh_temp.shape)
            # print(yh_temp[1,:])

            # JC notes:
            # yh_temp.shape[0] = num_samps
            # yh_temp.shape[1] = num_outputs+1

            if not idpt:
                # train theta normally (scaled velocities can depend on both input velocities)
                # Theta has the following form: [[w_xx, w_xy, b_x]  (actually transpose of this?)
                #                                [w_yx, w_yy, b_y]]
                theta = torch.mm(
                    torch.mm(
                        torch.pinverse(torch.mm(torch.t(yh_temp), yh_temp)),
                        torch.t(yh_temp),
                    ),
                    y,
                )
                print(theta) if verbose else _
            else:
                # train ~special~ theta
                # (scaled velocities are indpendent of each other - this is the typical method)
                # Theta has the following form: [[w_x,   0]
                #                                [0,   w_y]
                #                                [b_x, b_y]]
                theta = torch.zeros((num_outputs + 1, num_outputs)).to(
                    device=device, dtype=dtype
                )
                for i in range(num_outputs):
                    yhi = yh_temp[:, (i, -1)]
                    thetai = torch.matmul(
                        torch.mm(
                            torch.pinverse(torch.mm(torch.t(yhi), yhi)), torch.t(yhi)
                        ),
                        y[:, i],
                    )
                    theta[i, i] = thetai[0]  # gain
                    theta[-1, i] = thetai[1]  # bias
                    if subtract_median:  # use the median as the bias
                        theta[-1, i] = -1 * medians[i]
                    print(
                        "Finger %d RR Calculated Gain, Offset: %.6f, %.6f"
                        % (i, thetai[0], thetai[1])
                    ) if verbose else _
    return theta


class OutputScaler:
    def __init__(self, gains, biases, scaler_type=""):
        """An object to linearly scale data, like the output of a neural network

        Args:
            channel_gains (1d np array):  [1,NumOutputs] array of gains
            bias (1d np array):           [1,NumOutputs] array of biases
            scaler_type (str, optional): 'regression' or 'peaks' or 'noscale', etc.
        """
        self.gains = gains
        self.biases = biases
        self.scaler_type = scaler_type

    def scale(self, data):
        """data should be an numpy array/tensor of shape [N, NumOutputs]"""
        # TODO maybe add torch compatibility

        N = data.shape[0]
        scaled_data = np.tile(self.gains, (N, 1)) * (
            data + np.tile(self.biases, (N, 1))
        )
        # scaled_data = np.tile(self.gains, (N, 1)) * data + np.tile(self.biases, (N, 1))

        # scaled_data = np.zeros_like(data)
        # # print(self.gains)
        # # print(self.biases)
        # for i in range(data.shape[1]):
        #     scaled_data[:,i] = (data[:,i]+self.biases[0,i]) * self.gains[0,i]

        return scaled_data

        # Normal RR: we do y=mx+b, so gain first then bias


def generate_output_scaler(
    model, loader, num_outputs=2, scaler_type="regression", verbose=True
):
    """Returns a scaler object that scales the output of a decoder

    Args:
        model:      model
        loader:     dataloader
        num_chans:  how many channels (96)
        scaler_type:
            'regression':   linear regression between predicted and actual
            'peaks':        scale based on peaks (Willsey method)
            'noscale':      don't scale (just return the orignal predicted values)

    Returns:
        scaler: an OutputScaler object that takes returns scaled version of input data
    """

    gains = None
    biases = None

    if scaler_type == "regression":
        # fit constants using regression
        theta = calcGainRR(loader, model, idpt=True, verbose=verbose)
        # theta = [[w_x,   0]
        #          [0,   w_y]
        #          [b_x, b_y]]
        gains = np.zeros((1, num_outputs))
        biases = np.zeros((1, num_outputs))
        for i in range(num_outputs):
            gains[0, i] = theta[i, i]
            biases[0, i] = theta[num_outputs, i]

    elif scaler_type == "regressionmedian":
        # fit constants using regression
        theta = calcGainRR(
            loader, model, idpt=True, subtract_median=True, verbose=verbose
        )
        # theta = [[w_x,   0]
        #          [0,   w_y]
        #          [b_x, b_y]]
        gains = np.zeros((1, num_outputs))
        biases = np.zeros((1, num_outputs))
        for i in range(num_outputs):
            gains[0, i] = theta[i, i]
            biases[0, i] = theta[num_outputs, i]
        print(gains)
        print(biases)

    elif scaler_type == "peaks":
        # fit constants using peak detection
        ratios, medians = calc_gain_peaks(model, loader)
        gains = np.zeros((1, num_outputs))
        biases = np.zeros((1, num_outputs))
        for i in range(num_outputs):
            gains[0, i] = ratios[i]
            biases[0, i] = -1 * medians[i] * ratios[i]

    elif scaler_type == "noscale":
        # don't do any scaling
        gains = np.ones((1, num_outputs))
        biases = np.zeros((1, num_outputs))

    else:
        raise ValueError("unknown scaler type")

    return OutputScaler(gains, biases, scaler_type=scaler_type)


def save_nn_decoder(
    monkey,
    date,
    model,
    scaler,
    binsize,
    fingers,
    good_chans_SBP,
    training_inputs,
    training_outputs,
    fname_prefix="NN",
    alt_fpath=None,
    alt_fname=None,
    overwrite=False,
):
    """Saves a decoder, scaler, and training parameters to a file. Saves to monkey/date folder, unless
        an alternative filepath (alt_fpath) is given.

    Args:
        monkey (string):            'Joker'
        date (string):              '2022-01-01'
        model:                      decoder model
        scaler (OutputScaler):      scaler object for scaling the output
        binsize (int):              50
        fingers (list):             [2, 4] Note: this uses the matlab 1-indexing
        good_chans_SBP (list):      list of good channels
        training_inputs (dict):     dictionary of training inputs
        training_outputs (dict):    dictionary of training outputs
        fname_prefix (string, optional): 'NN' or 'RNN'
        alt_fpath (string, optional): alternative folder path to save the decoder
        alt_fname (string, optional): alternative decoder file name. Should end in '.pkl' (or whatever file extension desired)
        overwrite (bool, optional):   Default false, if so adds increasing number suffix to decoder name. Overwrites network if True and file exists.
    """

    # figure out folder path
    if alt_fpath is not None:
        fpath = alt_fpath
    else:
        fpath = os.path.join(get_server_data_path(), monkey, date)

    # figure out filename
    if alt_fname is not None:
        fname_out = alt_fname
    else:
        fname_out = f"decodeParams{fname_prefix}"

    # add suffix to name (determine what number decoder this should be)
    if not overwrite:
        count = 0
        # works fine for small numbers of files, will be inefficient if we're training like, 100 nns at a time
        while os.path.exists(
            os.path.join(fpath, f"{fname_out}{count}.pkl")
        ) or os.path.exists(os.path.join(fpath, f"{fname_out}{count}.mat")):
            count += 1
        fname_out = f"{fname_out}{count}.pkl"

    # save
    with open(os.path.join(fpath, fname_out), "wb") as f:
        pickle.dump(
            [
                model,
                scaler,
                binsize,
                fingers,
                good_chans_SBP,
                training_inputs,
                training_outputs,
            ],
            f,
        )
    print(f"decoder saved to: {os.path.join(fpath, fname_out)}")


def load_nn_decoder(monkey, date, decode_num=0, alt_fpath=None, alt_fname=None):
    """
    loads a decoder saved with save_nn_decoder, and returns preset variables.
    inputs:
        monkey (string):                    Name of the monkey (ex 'Joker)
        date (string):                      In the form 'YYYY-MM-DD'
        decode_num (int, optional):         default 0, decoder number to load if there are multiple in the same place
        alt_fpath (string/path, optional):  alternative folder path to the decoder
        alf_fname (string/path, optional):  alternative decoder file name. Should end in '.pkl'
    outputs:
        model:                      decoder model
        scaler (OutputScaler):      scaler object for scaling the output
        binsize (int):              50
        fingers (list):             [2, 4] Note: this uses the matlab 1-indexing
        good_chans_SBP (list):      list of good channels
        training_inputs (dict):     dictionary of training inputs
        training_outputs (dict):    dictionary of training outputs
    """
    # figure out folder path
    if alt_fpath is not None:
        fpath = alt_fpath
    else:
        fpath = os.path.join(get_server_data_path(), monkey, date)

    # filename
    if alt_fname is not None:
        fname = alt_fname
    else:
        fname = f"decodeParamsNN{decode_num}.pkl"

    # load
    with open(os.path.join(fpath, fname), "rb") as f:
        [
            model,
            scaler,
            binsize,
            fingers,
            good_chans_SBP,
            training_inputs,
            training_outputs,
        ] = pickle.load(f)

    return (
        model,
        scaler,
        binsize,
        fingers,
        good_chans_SBP,
        training_inputs,
        training_outputs,
    )


def load_model(monkey, date, model_fname, alt_fpath=None):
    """Loads a previously saved model. Used in ReFIT training.

    Args:
        monkey (string):            'Joker'
        date (string):              '2022-01-01'
        model_fname:                'NNDecoder0.pkl'
        alt_fpath (string, optional): alternative folder path for the decoder
    """
    if alt_fpath is not None:
        fpath = alt_fpath
    else:
        fpath = os.path.join(get_server_data_path(), monkey, date)

    with open(os.path.join(fpath, model_fname), "rb") as f:
        [model, scaler, binSize, fingers, good_chans_SBP, _, _] = pickle.load(f)
    return model


def reinit_model_weights(model):
    """Randomly re-initializes the weights of a model, useful for when training multiple models
    Args:
        model: input model
    Returns:
        model_out: a copy of the model with reset weights
    """
    model_out = copy.deepcopy(model)
    for layer in model_out.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
            # print(layer)
    return model_out


def train_N_models(
    model,
    num_models,
    optimizer_name,
    learning_rate,
    weight_decay,
    loss_func,
    loader_train,
    loader_val,
    check_accuracy_iters=100,
    verbose=True,
    scheduler_name=None,
    min_lr=None,
    scheduler_patience=None,
    max_iter=None,
    max_epoch=500,
    plot_progress=True,
    plot_simple=False,
    sparsify_model=False,
    sparse_pctiles=[],
    sparse_iters=[],
    bias_neural_std=None,
    noise_neural_std=None,
    noise_neural_walk_std=None,
):
    """Wrapper for the standard training function to allow training multiple models with different initializations.
        Note: all the inputs are the same as the 'train_model' function, except for 'num_models' and 'optimizer_name',
        'learning_rate', 'weight_decay', and 'scheduler_name'.

    Args:
        model:                          pytorch model
        num_models:                     how many different models to train
        optimizer_name:                 pytorch optimizer name(Adam)
        loss_func:                      loss func (nn.mseloss)
        learning_rate:                  learning rate for training
        weight_decay:                   optimizer weight decay
        loader_train ([Dataloader):     training laoder
        loader_val (Dataloader):        validation loader
        check_accuracy_iters (int):     [description]. Defaults to 100
        verbose (bool, optional):       if should print training updates
        scheduler_name (optional):           Learning rate scheduler NAME "Plateau". If provided, won't use the 'max_iter'
        min_lr (float, optional):       Training will stop when the scheduler reaches this learning rate
        scheduler_patience (int, optional): How many iters/100 to wait before reducing lr
        max_iter (int, optional):       Iteration to stop training. Used if no scheduler provided
        max_epoch (int, optional):      Max epoch. Will stop after this epoch.
        plot_progress (bool, optional): If should plot progress during training
        plot_simple (bool, optional):   True - use live in-depth plot. False - simple mse/corr plot.
        sparsify_model (bool, optional): If the model should be sparsified
        sparse_pctiles (list, optional): If sparsifying, what percentiles to use at each step. [60, 70, 80, 90, 95]
        sparse_iters (list, optional):  If sparsifying, which iter to sparsify at [2500, 2700, 2900, 3100, 3300]
        # TODO add noise descriptions

    Returns:
        [[model0, iter0, valloss0, corr0]...]:   trained model, final iteration, validation loss, validation correlation
    """
    results = []
    for i in range(num_models):

        # randomly reinit the model parameters before training
        thismodel = reinit_model_weights(model)

        # setup optimizers and scheduler
        if optimizer_name == "Adam":
            optimizer = optim.Adam(
                thismodel.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            raise ValueError("optimizer currently not supported")
        if scheduler_name is not None:
            if scheduler_name == "Plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=scheduler_patience,
                    verbose=verbose,
                )
            else:
                raise ValueError("scheduler currently not supported")
        else:
            scheduler = None

        theseresults = train_model(
            model=thismodel,
            optimizer=optimizer,
            loss_func=loss_func,
            loader_train=loader_train,
            loader_val=loader_val,
            check_accuracy_iters=check_accuracy_iters,
            verbose=verbose,
            scheduler=scheduler,
            min_lr=min_lr,
            max_iter=max_iter,
            max_epoch=max_epoch,
            plot_progress=plot_progress,
            plot_simple=plot_simple,
            sparsify_model=sparsify_model,
            sparse_pctiles=sparse_pctiles,
            sparse_iters=sparse_iters,
            bias_neural_std=bias_neural_std,
            noise_neural_std=noise_neural_std,
            noise_neural_walk_std=noise_neural_walk_std,
        )
        results.append(theseresults)

    return results


def kfTrain(X, Y, lag=0):
    # Confusing but X and Y must switch to be compatible with my KF code below. Sorry.
    X1 = Y
    # Ytrain = X[:,mask==1]
    Ytrain = X
    if lag > 0:
        Ytrain = Ytrain[:-lag, :]
        X1 = X1[lag:]  # Delay kinematics
        # Ytrain = Ytrain[lag:, :]    # Delay firing rates
        # X1 = Y[:-lag]
    N_train = Ytrain.shape[0]

    X1 = torch.cat((X1, torch.ones([X1.shape[0], 1]).to(dtype).to(device)), dim=1)

    Ytrain = torch.t(Ytrain)
    X1 = torch.t(X1)

    # Train A and C

    Xt = X1[:, 1:]
    Xtm1 = X1[:, 0:-1]

    XXm1 = torch.mm(Xt, torch.t(Xtm1))
    XX = torch.mm(Xtm1, torch.t(Xtm1))
    XXi = torch.pinverse(XX)
    A = XXm1.mm(XXi)

    YX = torch.mm(Ytrain, torch.t(X1))
    XX = torch.mm(X1, torch.t(X1))
    XXi = torch.pinverse(XX)
    C = torch.mm(YX, XXi)

    W = (1 / (N_train - 1)) * torch.mm(
        (Xt - torch.mm(A, Xtm1)), torch.t(Xt - torch.mm(A, Xtm1))
    )

    Q = (1 / N_train) * torch.mm(
        (Ytrain - torch.mm(C, X1)), torch.t(Ytrain - torch.mm(C, X1))
    )

    CtQinv = torch.mm(torch.t(C), torch.pinverse(Q))
    CtQinvC = torch.mm(torch.t(C), torch.mm(torch.pinverse(Q), C))

    params = {"A": A, "C": C, "W": W, "Q": Q, "CtQinv": CtQinv, "CtQinvC": CtQinvC}
    return params


def kfPredict(Xtest, params, initial_cond):
    A = torch.tensor(params["A"]).to(dtype).to(device)
    C = torch.tensor(params["C"]).to(dtype).to(device)
    W = torch.tensor(params["W"]).to(dtype).to(device)
    Q = torch.tensor(params["Q"]).to(dtype).to(device)

    Ytest = Xtest.T
    initial_cond = initial_cond.T

    # Initialize
    x = torch.cat(
        (initial_cond[:, 0:1], torch.ones([1, 1]).to(dtype).to(device)), dim=0
    )
    Pt = W

    N_test = Ytest.shape[1]

    # Iterate
    for k in np.arange(1, N_test):
        xtGtm1 = A.mm(x[:, -1:])
        PtGtm1 = A.mm(Pt).mm(A.T) + W
        Kt = torch.mm(PtGtm1.mm(C.T), torch.inverse(C.mm(PtGtm1).mm(C.T) + Q))
        xt = xtGtm1 + Kt.mm(Ytest[:, k : k + 1] - C.mm(x[:, -1:]))
        x = torch.cat([x, xt], axis=1)
        Pt = (torch.eye(Pt.shape[0]).to(dtype).to(device) - Kt.mm(C)).mm(PtGtm1)

    x = x.T

    yhat = x
    return yhat


"""
==========================================================
   LEGACY FUNCTIONS (PRE-2022)
   
   warning: functions probably aren't maintained
==========================================================
"""


def flatten(x, start_dim=1, end_dim=-1):
    return x.flatten(start_dim=start_dim, end_dim=end_dim)


def check_accuracy_part34(
    loader,
    model,
    input_opt=0,
    plotflag=False,
    best_val=-1,
    model_name="DefaultName",
    ConvSize=1,
    GF=1,
    offset=0,
    out="",
):
    if loader.dataset.train:
        print("Checking accuracy on validation set")
    else:
        print("Checking accuracy on test set")
    r_value = 0
    model.eval()  # set model to evaluation mode
    batches = len(list(loader))
    b, a = signal.cheby1(2, 0.1, 0.001, "hp", analog=False)
    # b, a = signal.cheby1(1, .01, [.001, .6], 'bandpass', analog=False)
    with torch.no_grad():
        for k1 in range(batches):
            temp = list(loader)
            x = temp[k1]["chans"]
            x = x[:, :, 0:ConvSize]
            y = temp[k1]["states"]
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.cpu()

            if ConvSize == 1:
                scores = model(x[:, :, 0:ConvSize])
            else:
                scores = model(x[:, :, 0:ConvSize])

            scores = GF * (scores.cpu().numpy() - offset)

            if False:
                for k3 in range(scores.shape[1]):
                    scores[:, k3] = signal.lfilter(b, a, scores[:, k3])

            peaks = np.zeros(scores.shape)
            noise_pow = np.zeros(scores.shape[1])
            avg_peak_control = np.zeros(scores.shape[1])
            SNR = np.zeros(scores.shape[1])

            noise_sig = np.zeros(y.shape[1])
            peaks_sig = np.zeros(y.shape)
            avg_peak_signal = np.zeros(y.shape[1])

            for k4 in range(scores.shape[1]):
                noise_pow[k4] = np.std(scores[:, k4])
                np_temp = 1.5 * np.median(np.abs(scores[:, k4]))
                peaks_ind, properties = find_peaks(
                    np.abs(scores[:, k4]), height=3 * np_temp
                )
                peaks[peaks_ind, k4] = 1
                SNR[k4] = np.mean(np.abs(scores[peaks_ind, k4])) / noise_pow[k4]
                avg_peak_control[k4] = np.mean(np.abs(scores[peaks_ind, k4]))

                np_temp = 1.5 * np.median(np.abs(y[:, k4]))
                peaks_ind_sig, properties = find_peaks(
                    np.abs(y[:, k4]), height=3 * np_temp
                )
                peaks_sig[peaks_ind_sig, k4] = 1
                avg_peak_signal[k4] = np.mean(np.abs(y[peaks_ind_sig, k4].numpy()))

            nos = scores.shape[1]
            for k in range(nos):
                norm_fact = np.sqrt(
                    np.correlate(
                        scores[:, k] - np.mean(scores[:, k]),
                        scores[:, k] - np.mean(scores[:, k]),
                    )
                    * np.correlate(
                        y[:, k] - torch.mean(y[:, k]), y[:, k] - torch.mean(y[:, k])
                    )
                )
                r_value += (
                    np.correlate(
                        scores[:, k] - np.mean(scores[:, k]),
                        y[:, k] - torch.mean(y[:, k]),
                    )
                    / norm_fact
                    / nos
                    / (batches)
                )
        if plotflag:
            for k2 in range(nos):
                norm_fact = np.sqrt(
                    np.correlate(
                        scores[:, k2] - np.mean(scores[:, k2]),
                        scores[:, k2] - np.mean(scores[:, k2]),
                    )
                    * np.correlate(
                        y[:, k2] - torch.mean(y[:, k2]), y[:, k2] - torch.mean(y[:, k2])
                    )
                )
                r_temp = (
                    np.correlate(
                        scores[:, k2] - np.mean(scores[:, k2]),
                        y[:, k2] - torch.mean(y[:, k2]),
                    )
                    / norm_fact
                )
                print("Correlation for Feature %1.0f is %g" % (k2, r_temp))
                print("Mean of feature is %g." % (np.mean(scores[:, k2])))
                print("Median of feature is %g." % (np.median(scores[:, k2])))
                print("SNR of feature is %g." % (SNR[k2]))
                print("Estimated gain is %g * dt." % (SNR[k2] / 3))
                plt.figure(figsize=(12, 9))
                times = np.arange(0, y.shape[0]).astype(float) / 20
                print(times.shape)
                print(y.shape)
                print(k2)
                plt.plot(times, y.cpu().detach().numpy()[:, k2])
                plt.plot(times, scores[:, k2])
                plt.plot(times, scores[:, k2] * peaks[:, k2], ".")
                plt.xlim(0, 10)
                plt.xlabel("Time (s)")
                plt.show()
        if best_val < r_value and input_opt != 0:
            best_val = r_value
            torch.save(model.state_dict(), out + model_name + ".pt")
            torch.save(
                {
                    "epoch": input_opt[2],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": input_opt[0].state_dict(),
                    "loss": input_opt[1],
                },
                out + model_name + "_withOptimizer.pt",
            )
        print("Mean Correlation is %g" % (r_value))
        print("Mean SNR is %g." % (np.mean(SNR)))
        print("Total number of peaks is %g." % (np.sum(peaks)))
        print("Average control peaks is %g." % (np.mean(avg_peak_control)))
        print("Average signal peaks is %g." % (np.mean(avg_peak_signal)))
        print(
            "Ratio of control to true signal is %g."
            % (np.mean(avg_peak_control) / (np.mean(avg_peak_signal)))
        )
        for k5 in range(nos):
            print("Feature number %g." % k5)
            print("Median of feature is %g." % (np.median(scores[:, k5])))
            print(
                "Ratio of control to true signal is %g."
                % (np.mean(avg_peak_control[k5]) / (np.mean(avg_peak_signal[k5])))
            )
            norm_fact = np.sqrt(
                np.correlate(
                    scores[:, k5] - np.mean(scores[:, k5]),
                    scores[:, k5] - np.mean(scores[:, k5]),
                )
                * np.correlate(
                    y[:, k5] - torch.mean(y[:, k5]), y[:, k5] - torch.mean(y[:, k5])
                )
            )
            print(
                "Correlation is %g"
                % (
                    np.correlate(
                        scores[:, k5] - np.mean(scores[:, k5]),
                        y[:, k5] - torch.mean(y[:, k5]),
                    )
                    / norm_fact
                )
            )

    return torch.tensor(r_value).to(dtype).to(device), best_val


def adjust_learning_rate(optimizer, lrd, epoch, schedule):
    """
    Multiply lrd to the learning rate if epoch is in schedule

    Inputs:
    - optimizer: An Optimizer object we will use to train the model
    - lrd: learning rate decay; a factor multiplied at scheduled epochs
    - epochs: the current epoch number
    - schedule: the list of epochs that requires learning rate update

    Returns: Nothing, but learning rate might be updated
    """
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            print(
                "lr decay from {} to {}".format(
                    param_group["lr"], param_group["lr"] * lrd
                )
            )
            param_group["lr"] *= lrd


def train_part345_online(
    model,
    optimizer,
    loader_input,
    epochs=1,
    e_in=0,
    loss=1e10,
    learning_rate_decay=0.1,
    schedule=[],
    verbose=True,
    model_name="DefaultName",
    ConvSize=1,
    zeroTrain=False,
    ReFIT=False,
    out="",
):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    loader_train = loader_input[0]
    loader_val = loader_input[1]

    print_every = 100

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    best_val = -1
    lossMSE = torch.nn.MSELoss()
    num_iters = epochs * len(loader_train)
    if verbose:
        num_prints = num_iters // print_every + 1
    else:
        num_prints = epochs
    acc_history = torch.zeros(num_prints, dtype=torch.float)
    iter_history = torch.zeros(num_prints, dtype=torch.long)
    for e in range(e_in, epochs):

        adjust_learning_rate(optimizer, learning_rate_decay, e, schedule)
        t = -1

        # this was your old code
        for batch in loader_train:

            t += 1

            x = batch["chans"]
            y = batch["states"]

            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            if ConvSize == 1:
                scores = model(x[:, :, 0:ConvSize])
            else:
                scores = model(x[:, :, 0:ConvSize])

            if ReFIT:
                indZero = scores * y > 0
                scores[indZero] = 0
                y[indZero] = 0

            # loss = lossMSE(scores[:, 0:1], y[:, 0:1])
            loss = lossMSE(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            tt = t + e * len(loader_train)

            if verbose and (
                tt % print_every == 0
                or (e == epochs - 1 and t == len(loader_train) - 1)
            ):
                torch.save(model.state_dict(), out + model_name + "_" + str(tt) + ".pt")
                print("Epoch %d, Iteration %d, loss = %.4f" % (e, tt, loss.item()))
                acc, best_val = check_accuracy_part34(
                    loader_val,
                    model,
                    input_opt=(optimizer, loss, e),
                    plotflag=False,
                    best_val=best_val,
                    model_name=model_name,
                    ConvSize=ConvSize,
                    out=out,
                )
                acc_history[tt // print_every] = acc
                iter_history[tt // print_every] = tt
                stheta = calcGainRR(loader=loader_val, model=model, ConvSize=ConvSize)
                print()
            elif not verbose and (t == len(loader_train) - 1):
                torch.save(model.state_dict(), out + model_name + "_" + str(tt) + ".pt")
                print("Epoch %d, Iteration %d, loss = %.4f" % (e, tt, loss.item()))
                acc, best_val = check_accuracy_part34(
                    loader_val,
                    model,
                    input_opt=(optimizer, loss, e),
                    plotflag=False,
                    best_val=best_val,
                    model_name=model_name,
                    ConvSize=ConvSize,
                    out=out,
                )
                acc_history[e] = acc
                iter_history[e] = tt
                stheta = calcGainRR(loader=loader_val, model=model, ConvSize=ConvSize)
                print()
    return acc_history, iter_history, best_val
