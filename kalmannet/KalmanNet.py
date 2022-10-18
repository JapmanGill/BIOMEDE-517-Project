"""# **Class: KalmanNet**"""

from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as func
from timeit import default_timer as timer


class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self, zero_hidden_state=False, nonlinear=False, B=None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.zero_hidden_state = zero_hidden_state
        self.nonlinear = nonlinear
        self.B = B

    #############
    ### Build ###
    #############
    def Build(self, A, C):

        self.InitSystemDynamics(A, C)

        # Number of neurons in the 1st hidden layer
        H1_KNet = (self.m + self.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (self.m * self.n) * 1 * (4)

        self.InitKGainNet(H1_KNet, H2_KNet)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.n  # x(t-1), y(t)

        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m + self.n * self.n) * 1
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

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
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)
        # Initialize GRU with kaiming instead of uniform initialization
        # for name, param in self.rnn_GRU.named_parameters():
        #     if "weight_ih" in name:
        #         torch.nn.init.kaiming_normal_(param.data, nonlinearity="relu")
        #     elif "weight_hh" in name:
        #         torch.nn.init.orthogonal_(param.data)
        #     elif "bias" in name:
        #         param.data.fill_(0)

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, A, C):
        # Set State Evolution Matrix
        self.A = A.to(self.device, non_blocking=True)
        self.A_T = torch.transpose(A, 0, 1)
        self.m = self.A.size()[0]

        # Set Observation Matrix
        self.C = C.to(self.device, non_blocking=True)
        self.C_T = torch.transpose(C, 0, 1)
        self.n = self.C.size()[0]

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, initial_state):
        # initial_state: (m,)
        self.x_prior = initial_state.to(self.device, non_blocking=True)
        self.x_posterior = initial_state.to(self.device, non_blocking=True)
        self.state_process_posterior_0 = initial_state.to(
            self.device, non_blocking=True
        )
        self.last_y = torch.zeros(self.n).to(self.device, non_blocking=True)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):

        # Compute the 1-st moment of x based on model knowledge and without process noise
        # self.state_process_prior_0 = torch.matmul(
        #     self.A, self.state_process_posterior_0
        # )

        # Compute the 1-st moment of y based on model knowledge and without noise
        # self.obs_process_0 = torch.matmul(self.C, self.state_process_prior_0)

        # Predict the 1-st moment of x
        self.x_prev_prior = self.x_prior
        self.x_prior = torch.matmul(self.A.float(), self.x_posterior)

        # Predict the 1-st moment of y
        if self.nonlinear:
            self.m1y = self.h(self.x_prior, self.B)
        else:
            # C comes with a bias column, so we need to add a 1 to the state vector
            x_prior_bias = torch.ones(self.m + 1)
            x_prior_bias[: self.m] = self.x_prior
            self.m1y = torch.matmul(self.C.float(), x_prior_bias)

    def h(self, x, B):
        x_vec = torch.empty(x.shape[0] + 1, 1).float().to(self.device)
        x_vec[0:2] = x[0:2, :]
        x_vec[2] = torch.sqrt(x[0] ** 2 + x[1] ** 2)
        return torch.matmul(x_vec.t(), B).t().to(self.device)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # Reshape and Normalize the difference in X prior
        # Featture 4: x_t|t - x_t|t-1
        # dm1x = self.x_prior - self.state_process_prior_0
        dm1x = self.x_posterior - self.x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 2: yt - y_t+1|t
        dm1y = y - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)

        # Feature 1: yt-yt-1
        # Won't normalize as we expect it to be pre-normalized
        # dm1y_prev = y - self.last_y
        # dm1y_norm_prev = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)
        # self.last_y = y

        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)

        # Kalman Gain Network Step
        start = timer()
        KG = self.KGain_step(KGainNet_in)
        end = timer()
        # print(f"KGain_step: {end - start}")
        # FIXME:
        # Only scale the velocity components
        # FIXME: FIXME:
        # KG[0, 24:] = KG[0, 24:] / 1000
        KG = KG / 10000

        # Reshape Kalman Gain to a Matrix
        self.k_gain = torch.reshape(KG, (self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        y_obs = y
        dy = y_obs - self.m1y

        # Compute the 1-st posterior moment
        innovation = torch.matmul(self.k_gain.float(), dy.float())
        self.x_posterior = self.x_prior + innovation

        # return
        return torch.squeeze(self.x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in.float())
        La1_out = self.KG_relu1(L1_out)

        ###########
        ### GRU ###
        ###########
        start = timer()
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(
            self.device, non_blocking=True
        )
        GRU_in[0, 0, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn.float())
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))
        end = timer()
        # print(f"GRU: {end - start}")

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.KG_l3(La2_out)
        return L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        # yt must be: (n,)
        yt = yt.to(self.device, non_blocking=True)
        return self.KNet_step(yt)

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
            self.InitSequence(x_0[b, :])
            x_out[b, :, :] = self.forward_sequence(y_batch[b, :, :])
        return x_out

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data
