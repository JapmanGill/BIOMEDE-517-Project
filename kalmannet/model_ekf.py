#%%
import math
import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
from torch import autograd
import sys
from parameters_ekf_fingflex import A, B, Q, W

if torch.cuda.is_available():
    dev = torch.device(
        "cuda:0"
    )  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

#%%
def f(x):
    return torch.matmul(A, x).to(dev)


def h(x):
    x_vec = torch.empty(x.shape[0] + 2).float().to(dev)
    x_vec[0:2] = x[0:2]
    x_vec[2] = torch.sqrt(x[0] ** 2 + x[1] ** 2)
    x_vec[3:5] = x[2:4]
    x_vec[5] = torch.sqrt(x[2] ** 2 + x[3] ** 2)

    return torch.matmul(x_vec, B).to(dev)
