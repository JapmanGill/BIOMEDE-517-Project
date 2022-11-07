"""# **Class: KalmanNet**"""

from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as func
from timeit import default_timer as timer


class KalmanNetNN(torch.nn.Module):
    def __init__(
        self,
        bin_size,
        zero_hidden_state=False,
        nonlinear=False,
        non_linear_model=None,
        reg_kf=False,
        h1_size=None,
        h2_size=None,
        hidden_dim=None,
        gain_scaler=10000,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.zero_hidden_state = zero_hidden_state
        self.nonlinear = nonlinear
        self.non_linear_model = non_linear_model
        self.reg_kf = reg_kf

        self.h1_size = h1_size
        self.h2_size = h2_size
        self.hidden_dim = hidden_dim
        self.gain_scaler = gain_scaler
        self.bin_size = bin_size

    #############
    ### Build ###
    #############
    def build(self, A, C, W=None, Q=None):

        self.init_system_dynamics(A, C, W, Q)

        # Number of neurons in the 1st hidden layer
        h1_size = (
            self.h1_size if self.h1_size is not None else (self.m + self.n) * (10) * 8
        )

        # Number of neurons in the 2nd hidden layer
        h2_size = (
            self.h2_size if self.h2_size is not None else (self.m * self.n) * 1 * (4)
        )

        self.init_kgain_net(h1_size, h2_size)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def init_kgain_net(self, h1_size, h2_size):
        # Input Dimensions
        in_dim = self.m + self.n  # x(t-1), y(t)

        # Output Dimensions
        out_dim = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.linear1 = torch.nn.Linear(in_dim, h1_size, bias=True).to(self.device)

        # ReLU (Rectified Linear Unit) Activation Function
        self.relu1 = torch.nn.ReLU().to(self.device)

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = h1_size
        # Hidden Dimension
        if self.hidden_dim is None:
            self.hidden_dim = (self.m * self.m + self.n * self.n) * 1
        # Number of Layers
        self.n_layers = 1
        # Batch Size. Can't be greater than 1, as we can't have batches of sequences
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # Initialize a Tensor for Hidden State
        if self.zero_hidden_state:
            self.hn = torch.zeros(
                self.seq_len_hidden, self.batch_size, self.hidden_dim
            ).to(self.device, non_blocking=True)
        else:
            self.hn = torch.randn(
                self.seq_len_hidden, self.batch_size, self.hidden_dim
            ).to(self.device, non_blocking=True)

        # Initialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers).to(
            self.device
        )
        # Initialize GRU with kaiming instead of uniform initialization
        # for name, param in self.rnn_GRU.named_parameters():
        #     if "weight_ih" in name:
        #         torch.nn.init.kaiming_normal_(param.data, nonlinearity="relu")
        #     elif "weight_hh" in name:
        #         torch.nn.init.orthogonal_(param.data)
        #     elif "bias" in name:
        #         param.data.fill_(0)

        # Hidden layer
        self.linear2 = torch.nn.Linear(self.hidden_dim, h2_size, bias=True).to(
            self.device
        )

        # ReLU (Rectified Linear Unit) Activation Function
        self.relu2 = torch.nn.ReLU().to(self.device)

        # Output layer
        self.linear3 = torch.nn.Linear(h2_size, out_dim, bias=True).to(self.device)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def init_system_dynamics(self, A, C, W, Q):
        # Set State Evolution Matrix
        self.A = A.to(self.device, non_blocking=True).float()
        self.A_T = torch.transpose(A, 0, 1).to(self.device, non_blocking=True).float()
        self.m = self.A.size()[0]

        # Set Observation Matrix
        self.C = C.to(self.device, non_blocking=True).float()
        self.C_T = torch.transpose(C, 0, 1).to(self.device, non_blocking=True).float()
        self.n = self.C.size()[0]

        # Noise covariances
        self.W = W.to(self.device, non_blocking=True).float() if W is not None else None
        self.Q = Q.to(self.device, non_blocking=True).float() if Q is not None else None

    # Initialize sequence
    def init_sequence(self, initial_state):
        # initial_state: (m,)
        self.x_prior = initial_state.to(self.device, non_blocking=True)
        self.x_posterior = initial_state.to(self.device, non_blocking=True)
        self.last_y = torch.zeros(self.n).to(self.device, non_blocking=True)

        self.P = torch.clone(self.W) if self.W is not None else None

    # Priors
    def step_prior(self):
        # Prior state
        self.x_prev_prior = self.x_prior
        self.x_prior = torch.matmul(self.A.float(), self.x_posterior)
        # TODO: compute prior estimate of P if running regular KF

        # Predict the 1-st moment of y
        if self.nonlinear:
            self.y_prior = self.h(self.x_prior, self.B)
        else:
            # TODO: auto detect if C has bias column
            # C comes with a bias column, so we need to add a 1 to the state vector
            # x_prior_bias = torch.ones(self.m + 1).to(device=self.device)
            # x_prior_bias[: self.m] = self.x_prior
            self.y_prior = torch.matmul(self.C.float(), self.x_prior)

    # Non linear function
    def h(self, x, B):
        x_vec = torch.empty(x.shape[0] + 1, 1).float().to(self.device)
        x_vec[0:2] = x[0:2, :]
        x_vec[2] = torch.sqrt(x[0] ** 2 + x[1] ** 2)
        return torch.matmul(x_vec.t(), B).t().to(self.device)

    # Kalman gain estimation
    def step_KGain_est(self, y):
        # TODO: compute all features
        # Reshape and Normalize the difference in X prior
        # Feature 4: x_t|t - x_t|t-1
        # f4 = self.x_prior - self.state_process_prior_0
        # TODO: fix. Need the old posterior and prior
        f4 = self.x_posterior - self.x_prior
        f4_reshape = torch.squeeze(f4)
        f4_norm = func.normalize(f4_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 2: yt - y_t+1|t
        f2 = y - torch.squeeze(self.y_prior)
        f2_norm = func.normalize(f2, p=2, dim=0, eps=1e-12, out=None)

        # KGain Net Input
        KGainNet_in = torch.cat([f2_norm, f4_norm], dim=0)

        # Kalman Gain Network Step
        start = timer()
        KG = self.KGain_step(KGainNet_in)
        end = timer()
        # print(f"KGain_step: {end - start}")
        # FIXME: there must be a better way to do this
        KG = KG / self.gain_scaler

        # Reshape Kalman Gain to a Matrix and return
        return torch.reshape(KG, (self.m, self.n))

    # Full KF step
    def kf_step(self, y):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        if self.reg_kf:
            self.k_gain = self.kgain_regkf()
        else:
            self.k_gain = self.step_KGain_est(y)

        # Innovation
        y_obs = y
        dy = y_obs - self.y_prior

        # Compute the 1-st posterior moment
        innovation = torch.matmul(self.k_gain.float(), dy.float())
        self.x_posterior = self.x_prior + innovation

        # return
        return torch.squeeze(self.x_posterior)

    # Kalman Gain step
    def KGain_step(self, network_input):

        # Input layer
        l1_out = self.linear1(network_input.float())
        la1_out = self.relu1(l1_out)

        # GRU
        start = timer()
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(
            self.device, non_blocking=True
        )
        GRU_in[0, 0, :] = la1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn.float())
        GRU_out = torch.reshape(GRU_out, (1, self.hidden_dim))
        end = timer()
        # print(f"GRU: {end - start}")

        # Hidden layer
        l2_out = self.linear2(GRU_out)
        la2_out = self.relu2(l2_out)

        # Output layer
        l3_out = self.linear3(la2_out)
        return l3_out

    def kgain_regkf(self):
        self.P = self.A @ self.P @ self.A_T + self.W
        k_gain = self.P @ self.C_T @ torch.inverse(self.C @ self.P @ self.C_T + self.Q)
        self.P = (torch.eye(self.m).to(self.device) - k_gain @ self.C) @ self.P
        return k_gain

    # Forward
    def forward(self, yt):
        # yt must be: (n,)
        yt = yt.to(self.device, non_blocking=True)
        return self.kf_step(yt)

    def forward_sequence(self, y_seq):
        # y_seq must be: (n, seq_len)
        y_seq = y_seq.to(self.device, non_blocking=True)
        x_out = torch.empty(self.m, y_seq.shape[1])
        for t in range(y_seq.shape[1]):
            x_out[:, t] = self.forward(y_seq[:, t])
        return x_out

    def forward_batch(self, y_batch, x_0):
        # y_batch must be: (batch_size, n, seq_len)
        y_batch = y_batch.to(self.device, non_blocking=True)
        x_out = torch.empty(y_batch.shape[0], self.m, y_batch.shape[2])
        for b in range(y_batch.shape[0]):
            x_out[b, :, 0] = x_0[b, :]
            self.init_sequence(x_0[b, :])
            # Initialize sequence and then pass forward the rest of it
            x_out[b, :, 1:] = self.forward_sequence(y_batch[b, :, 1:])
        return x_out

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data
