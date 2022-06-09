import torch
import torch.nn as nn
import random
import time
from Plot import Plot
import numpy as np
import wandb


class Pipeline_KF:
    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + "/"
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName
        self.PipelineName = self.folderName + "pipeline_" + self.modelName

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch  # Number of Samples in Batch
        self.learningRate = learningRate  # Learning Rate
        self.weightDecay = weightDecay  # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction="mean")

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay
        )

    def NNTrain(
        self,
        n_Examples,
        train_input,
        train_target,
        train_x0,
        n_CV,
        cv_input,
        cv_target,
        cv_x0,
    ):

        self.N_E = n_Examples
        self.N_CV = n_CV

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs])
        corr_cv_batch = torch.empty([self.N_CV])
        self.corr_cv_epoch = torch.empty([self.N_Epochs])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])
        corr_train_batch = torch.empty([self.N_CV])
        self.corr_train_epoch = torch.empty([self.N_Epochs])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_Epochs):
            print(f"Epoch {ti+1}...")

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            print("Cross validation...")

            for j in range(0, self.N_CV):
                y_cv = cv_input[j, :, :]
                self.model.InitSequence(cv_x0[j, :])

                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T_val)
                for t in range(0, self.ssModel.T_val):
                    x_out_cv[:, t] = self.model(y_cv[:, t].float())

                # Compute Training Loss
                MSE_cv_linear_batch[j] = self.loss_fn(
                    x_out_cv, cv_target[j, :, :]
                ).item()
                # Compute correlation
                corr = np.zeros(4)
                for i in range(4):
                    corr[i] = np.corrcoef(
                        x_out_cv.detach().cpu(), cv_target[j, :, :].cpu()
                    )[0, 1]
                corr_cv_batch[j] = np.mean(corr)

            # Average
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
            self.corr_cv_epoch[ti] = torch.mean(corr_cv_batch)

            if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()
            print("Training...")

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                n_e = random.randint(0, self.N_E - 1)

                y_training = train_input[n_e, :, :]
                self.model.InitSequence(train_x0[n_e, :])

                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T)
                for t in range(0, self.ssModel.T):
                    x_out_training[:, t] = self.model(y_training[:, t])

                # Compute Training Loss
                LOSS = self.loss_fn(x_out_training, train_target[n_e, :, :])
                # Compute correlation
                corr = np.zeros(4)
                for i in range(4):
                    corr[i] = np.corrcoef(
                        x_out_training[i, :].detach().cpu(), train_target[i, :].cpu()
                    )[0, 1]
                MSE_train_linear_batch[j] = LOSS.item()
                corr_train_batch[j] = np.mean(corr)

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(
                self.MSE_train_linear_epoch[ti]
            )
            self.corr_train_epoch[ti] = torch.mean(corr_train_batch)

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            print(f"Epoch {ti+1}")
            print(
                f"MSE_train (dB):{self.MSE_train_dB_epoch[ti]:.3f}\tCorr_train: {self.corr_train_epoch[ti]:.3f}"
            )
            print(
                f"MSE_cv (dB):{self.MSE_cv_dB_epoch[ti]:.3f}\tCorr_cv: {self.corr_cv_epoch[ti]:.3f}"
            )
            log_dict = {
                "mse_train_db": self.MSE_train_dB_epoch[ti],
                "corr_train": self.corr_train_epoch[ti],
                "mse_cv_db": self.MSE_cv_dB_epoch[ti],
                "corr_cv": self.corr_cv_epoch[ti],
            }

            if ti >= 1:
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_corr_train = self.corr_train_epoch[ti] - self.corr_train_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                d_corr_cv = self.corr_cv_epoch[ti] - self.corr_cv_epoch[ti - 1]
                log_dict.update(
                    {
                        "d_train_db": d_train,
                        "d_train_corr": d_corr_train,
                        "d_cv_db": d_cv,
                        "d_cv_corr": d_corr_cv,
                    }
                )
                print("Differences")
                print(
                    f"\tDiff MSE_train (dB): {d_train:.3f}\tDiff corr_train: {d_corr_train:.3f}"
                )
                print(f"\tDiff MSE_cv (dB): {d_cv:.3f}\tDiff corr_cv: {d_corr_cv:.3f}")

            print(f"Optimal epoch {self.MSE_cv_idx_opt+1}")
            print(
                f"\tMSE_cv (dB): {self.MSE_cv_dB_epoch[self.MSE_cv_idx_opt]:.3f}\tCorr_cv: {self.corr_cv_epoch[self.MSE_cv_idx_opt]:.3f}"
            )
            log_dict.update(
                {
                    "optimal_epoch": self.MSE_cv_idx_opt + 1,
                    "current_epoch": ti + 1,
                    "optimal_cv_mse_db": self.MSE_cv_dB_epoch[self.MSE_cv_idx_opt],
                    "optimal_cv_corr": self.corr_cv_epoch[self.MSE_cv_idx_opt],
                }
            )
            wandb.log(log_dict)

    def NNTest(self, n_Test, test_input, test_target, x_0):

        self.N_T = n_Test

        self.MSE_test_linear_arr = torch.empty([self.N_T])
        self.corr_test = torch.empty([self.N_T])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction="mean")

        self.model = torch.load(self.modelFileName)

        self.model.eval()

        torch.no_grad()

        start = time.time()

        for j in range(0, self.N_T):

            y_mdl_tst = test_input[j, :, :]

            self.model.InitSequence(x_0)

            x_out_test = torch.empty(self.ssModel.m, self.ssModel.T_test)

            for t in range(0, self.ssModel.T_test):
                x_out_test[:, t] = self.model(y_mdl_tst[:, t])

            self.MSE_test_linear_arr[j] = loss_fn(
                x_out_test, test_target[j, :, :]
            ).item()
            # Compute correlation
            corr = np.zeros(4)
            for i in range(4):
                corr[i] = np.corrcoef(
                    x_out_test.detach().cpu(), test_target[j, i, :].cpu()
                )[0, 1]
            self.corr_test[j] = np.mean(corr)

        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_dB_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_dB_std)

        # Correlation
        corr_mean = torch.mean(self.corr_test)

        log_dict = {
            "mse_test_db_avg": self.MSE_test_dB_avg,
            "corr_test_avg": corr_mean,
        }
        wandb.log(log_dict)

        print(f"MSE_test (dB):{self.MSE_test_dB_avg:.3f}\tCorr_test: {corr_mean:.3f}")
        # Print MSE Cross Validation
        # str = self.modelName + "-" + "MSE Test:"
        # print(str, self.MSE_test_dB_avg, "[dB]")
        # str = self.modelName + "-" + "STD Test:"
        # print(str, self.MSE_test_dB_std, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        # return [
        #     self.MSE_test_linear_arr,
        #     self.MSE_test_linear_avg,
        #     self.MSE_test_dB_avg,
        #     x_out_test,
        # ]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(
            self.N_Epochs,
            MSE_KF_dB_avg,
            self.MSE_test_dB_avg.cpu(),
            self.MSE_cv_dB_epoch.cpu(),
            self.MSE_train_dB_epoch.cpu(),
        )

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)
