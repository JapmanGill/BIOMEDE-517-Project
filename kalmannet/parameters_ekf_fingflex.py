#%%import torch
import math
import scipy.io as scio
import torch

if torch.cuda.is_available():
    cuda0 = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    cuda0 = torch.device("cpu")
    print("Running on the CPU")

#%%
matA = scio.loadmat("data/finger_flexion/A.mat")
matB = scio.loadmat("data/finger_flexion/B.mat")
matQ = scio.loadmat("data/finger_flexion/Q.mat")
matW = scio.loadmat("data/finger_flexion/W.mat")

A = torch.tensor(matA["A"]).float()
B = torch.tensor(matB["B"]).float()
Q = torch.tensor(matQ["Q"]).float()
W = torch.tensor(matW["W"]).float()
