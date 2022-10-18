import torch
import torch.nn as nn
import random
import time
from Plot import Plot
import numpy as np
import wandb
from utils.utils import compute_correlation
from prettytable import PrettyTable


class Pipeline_KF:
    def __init__(self, folder_name, modelName, good_chans, pred_type="v"):
        super().__init__()
        self.folder_name = folder_name + "/"
        self.modelName = modelName
        self.modelFileName = self.folder_name + "model_" + self.modelName
        self.PipelineName = self.folder_name + "pipeline_" + self.modelName
        self.good_chans = torch.tensor(good_chans, dtype=torch.int)
        self.pred_type = pred_type

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

    def new_train(self, train_dataloader, cv_dataloader, compute_val_every=100):
        optimal_mse = 1000
        optimal_correlation = torch.empty([self.ssModel.m])
        optimal_epoch = -1

        self.count_parameters(self.model)
        for epoch in range(self.N_Epochs):
            print(f"Epoch {epoch+1}...")
            print("Training...")

            if self.pred_type == "v":
                train_loss = torch.empty([len(train_dataloader), 2])
                train_corr = torch.empty([len(train_dataloader), 2])
            else:
                train_loss = torch.empty([len(train_dataloader), 4])
                train_corr = torch.empty([len(train_dataloader), 4])
            for i, loader_dict in enumerate(train_dataloader):
                # Training
                self.model.train()
                num_iteration = epoch * len(train_dataloader) + i + 1
                y = loader_dict["chans"]
                # Remove bad channels
                y = torch.index_select(y, 1, self.good_chans)
                x = loader_dict["states_hist"]
                x_0 = loader_dict["initial_states"]
                # Initialize hidden state: necessary for backprop
                # FIXME: initialize one hidden state per sequence
                self.model.init_hidden()
                self.optimizer.zero_grad()
                # Initialize sequence for KalmanFilter
                # self.model.InitSequence(x_0[0, :])
                # Do forward pass of full batch
                x_hat = self.model.forward_batch(y, x_0)
                # Compute MSE loss
                if self.pred_type == "v":
                    loss = self.loss_fn(x_hat[:, 2:, :], x[:, 2:, :])
                    train_loss[i, :] = (
                        ((x_hat[:, 2:, :] - x[:, 2:, :]) ** 2)
                        .mean(axis=[0, 2])
                        .detach()
                    )
                else:
                    loss = self.loss_fn(x_hat, x)
                    train_loss[i, :] = ((x_hat - x) ** 2).mean(axis=[0, 2]).detach()
                # Backpropagate and update weights
                loss.backward()

                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)

                self.optimizer.step()
                # Compute correlation between x and x_hat
                if self.pred_type == "v":
                    corr = compute_correlation(
                        x[:, 2:, :].detach().cpu(), x_hat[:, 2:, :].detach().cpu()
                    )
                else:
                    corr = compute_correlation(x.detach().cpu(), x_hat.detach().cpu())

                train_corr[i, :] = torch.nanmean(torch.from_numpy(corr), 0)
                print("Training", loss.item(), np.nanmean(corr, 0))

                log_dict = {
                    "iteration": num_iteration,
                    "train_loss": loss.item(),
                    "train_corr": np.nanmean(corr, 0),
                    "mean_train_corr": np.nanmean(corr),
                }

                # TODO: Every N iterations, run model on validation set
                if num_iteration % compute_val_every == 0:
                    self.model.eval()
                    print("Validating...")
                    if self.pred_type == "v":
                        val_loss = torch.empty([len(cv_dataloader), 2])
                        val_corr = np.zeros([len(cv_dataloader), 2])
                    else:
                        val_loss = torch.empty([len(cv_dataloader), 4])
                        val_corr = np.zeros([len(cv_dataloader), 4])
                    for j, loader_dict in enumerate(cv_dataloader):
                        y = loader_dict["chans_nohist"]
                        # Remove bad channels
                        y = torch.index_select(y, 1, self.good_chans)
                        x = loader_dict["states"]
                        x_0 = loader_dict["initial_states"]
                        # Initialize hidden state: necessary for backprop
                        self.model.init_hidden()
                        # Initialize sequence for KalmanFilter
                        self.model.InitSequence(x_0[0, :])
                        # Run model on validation set
                        # Output is (seq_len, m)
                        x_hat = self.model.forward_sequence(y.T).T
                        # Compute MSE loss and correlation
                        if self.pred_type == "v":
                            val_loss[j, :] = (
                                ((x_hat[:, 2:] - x[:, 2:]) ** 2).mean(axis=[0]).detach()
                            )
                            val_corr[j, :] = compute_correlation(
                                x[:, 2:].detach().cpu().T,
                                x_hat[:, 2:].detach().cpu().T,
                            )
                        else:
                            val_loss[j, :] = ((x_hat - x) ** 2).mean(axis=[0]).detach()
                            val_corr[j, :] = compute_correlation(
                                x.detach().cpu().T,
                                x_hat.detach().cpu().T,
                            )
                    log_dict.update(
                        {"val_loss": val_loss.mean(), "val_corr": val_corr.mean()}
                    )
                    print(
                        f"Validation loss: {val_loss.mean()}, corr: {val_corr.mean()}"
                    )

                # Log to Wandb
                wandb.log(log_dict)

            # log_dict = {
            #     "mse_train_all": np.nanmean(train_loss, 0).tolist(),
            #     "mse_train": np.nanmean(train_loss),
            #     "corr_train_all": np.nanmean(train_corr, 0).tolist(),
            #     "corr_train": np.nanmean(train_corr),
            #     "epoch": epoch,
            # }
            # wandb.log(log_dict)

            # Validation
            self.model.eval()
            print("Validation...")

            val_loss = torch.empty([len(cv_dataloader), self.ssModel.m])
            val_corr = torch.empty([len(cv_dataloader), self.ssModel.m])
            for i, loader_dict in enumerate(cv_dataloader):
                y = loader_dict["chans_nohist"]
                # Remove bad channels
                y = torch.index_select(y, 1, self.good_chans).T
                x = loader_dict["states"].T
                x_0 = loader_dict["initial_states"]
                # Initialize sequence for KalmanFilter
                self.model.InitSequence(x_0[0, :])
                # Do forward pass of full batch
                x_hat = self.model.forward_sequence(y)
                # Compute MSE loss
                loss = self.loss_fn(x_hat, x)
                val_loss[i, :] = ((x_hat - x) ** 2).mean(axis=[0, 1]).detach()
                # Compute correlation between x and x_hat
                corr = compute_correlation(x.detach().cpu(), x_hat.detach().cpu())
                val_corr[i, :] = torch.from_numpy(corr)

            mse = np.nanmean(val_loss.cpu())
            print("Validation", mse, np.nanmean(val_corr.cpu(), 0).tolist())
            if mse < optimal_mse:
                optimal_mse = mse
                optimal_correlation = np.nanmean(val_corr.cpu(), 0).tolist()
                optimal_epoch = epoch
                torch.save(self.model, self.modelFileName)
                print(f"Saving model with MSE {optimal_mse}")

            # log_dict.update(
            #     {
            #         "mse_cv_all": np.nanmean(val_loss, 0).tolist(),
            #         "corr_cv_all": np.nanmean(val_corr, 0).tolist(),
            #         "mse_cv": np.nanmean(val_loss),
            #         "corr_cv": np.nanmean(val_corr),
            #         "optimal_mse_cv": optimal_mse,
            #         "optimal_correlation_cv": optimal_correlation,
            #         "optimal_epoch": optimal_epoch,
            #     }
            # )
            # wandb.log(log_dict)

    def count_parameters(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def NNTrain(
        self, n_Examples, train_dataloader, n_CV, cv_dataloader, only_vel=False
    ):

        self.N_E = n_Examples
        self.N_CV = n_CV

        MSE_cv_batch = torch.empty([self.N_CV])
        self.MSE_cv_epoch = torch.empty([self.N_Epochs])
        corr_cv_batch = torch.empty([self.N_CV])
        self.corr_cv_epoch = torch.empty([self.N_Epochs])

        MSE_train_batch = torch.empty([self.N_B])
        MSE_train_batch_epoch = torch.empty([len(train_dataloader)])
        self.MSE_train_epoch = torch.empty([self.N_Epochs])
        corr_train_batch = torch.empty([self.N_B])
        corr_train_batch_epoch = torch.empty([len(train_dataloader)])
        self.corr_train_epoch = torch.empty([self.N_Epochs])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_Epochs):
            print(f"Epoch {ti+1}...")

            # Training
            self.model.train()
            # Init Hidden State
            for i, (y, x_target, x_0) in enumerate(train_dataloader):
                loss_sum = 0
                if i == 0:
                    self.model.InitSequence(x_0[0, :])
                # Iterate over the first dimension of y (the batch dimension)
                self.optimizer.zero_grad()
                for j in range(0, y.shape[0]):
                    self.model.init_hidden()
                    y_batch, x_batch, x_0_batch = (
                        y[j, :, :],
                        x_target[j, :, :],
                        x_0[j, :],
                    )

                    # # Initialize sequence
                    # self.model.InitSequence(x_0_batch)
                    # Iterate over the sequence
                    x_out_training = self.model.forward_sequence(y_batch)
                    if only_vel:
                        loss = self.loss_fn(x_out_training[2:, :], x_batch[2:, :])
                        loss.backward()
                    else:
                        loss = self.loss_fn(x_out_training, x_batch)
                        loss.backward()

                    MSE_train_batch[j] = loss.item()

                    # Compute correlation
                    corr = compute_correlation(
                        x_out_training.detach().cpu(), x_batch.cpu()
                    )
                    if only_vel:
                        corr_train_batch[j] = np.mean(corr[2:])
                    else:
                        corr_train_batch[j] = np.mean(corr)
                    del loss
                # Backpropagation (update gradients)
                # loss_sum = loss_sum / y.shape[0]
                # loss_sum.backward()
                self.optimizer.step()
                # Update weights once per every batch
                # self.optimizer.step()
                MSE_train_batch_epoch[i] = torch.nanmean(MSE_train_batch)
                corr_train_batch_epoch[i] = torch.nanmean(corr_train_batch)
                print(torch.mean(MSE_train_batch), torch.nanmean(corr_train_batch))

            # Average
            self.MSE_train_epoch[ti] = torch.mean(MSE_train_batch_epoch)
            self.corr_train_epoch[ti] = torch.mean(corr_train_batch)

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            print("Cross validation...")

            for j, (cv_input, cv_target, cv_x0) in enumerate(cv_dataloader):
                y_cv = torch.squeeze(cv_input)
                self.model.InitSequence(cv_x0[0, :, :])

                x_out_cv = self.model.forward_sequence(y_cv.float())

                # Compute CV Loss
                if only_vel:
                    MSE_cv_batch[j] = self.loss_fn(x_out_cv[2:, :], cv_target[0, 2:, :])
                else:
                    MSE_cv_batch[j] = self.loss_fn(
                        x_out_cv, torch.squeeze(cv_target)
                    ).item()
                # Compute correlation
                corr = compute_correlation(x_out_cv.detach().cpu(), cv_target.cpu())
                if only_vel:
                    corr_cv_batch[j] = np.mean(corr[2:])
                else:
                    corr_cv_batch[j] = np.mean(corr)

            # Average
            self.MSE_cv_epoch[ti] = torch.nanmean(MSE_cv_batch)
            self.corr_cv_epoch[ti] = torch.nanmean(corr_cv_batch)

            if self.MSE_cv_epoch[ti] < self.MSE_cv_opt:
                self.MSE_cv_opt = self.MSE_cv_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)

            ########################
            ### Training Summary ###
            ########################
            print(f"Epoch {ti+1}")
            print(
                f"MSE_train:{self.MSE_train_epoch[ti]:.3f}\tCorr_train: {self.corr_train_epoch[ti]:.3f}"
            )
            print(
                f"MSE_cv:{self.MSE_cv_epoch[ti]:.3f}\tCorr_cv: {self.corr_cv_epoch[ti]:.3f}"
            )
            log_dict = {
                "mse_train": self.MSE_train_epoch[ti],
                "corr_train": self.corr_train_epoch[ti],
                "mse_cv": self.MSE_cv_epoch[ti],
                "corr_cv": self.corr_cv_epoch[ti],
            }

            if ti >= 1:
                d_train = self.MSE_train_epoch[ti] - self.MSE_train_epoch[ti - 1]
                d_corr_train = self.corr_train_epoch[ti] - self.corr_train_epoch[ti - 1]
                d_cv = self.MSE_cv_epoch[ti] - self.MSE_cv_epoch[ti - 1]
                d_corr_cv = self.corr_cv_epoch[ti] - self.corr_cv_epoch[ti - 1]
                log_dict.update(
                    {
                        "d_train": d_train,
                        "d_train_corr": d_corr_train,
                        "d_cv": d_cv,
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
                f"\tMSE_cv (dB): {self.MSE_cv_epoch[self.MSE_cv_idx_opt]:.3f}\tCorr_cv: {self.corr_cv_epoch[self.MSE_cv_idx_opt]:.3f}"
            )
            log_dict.update(
                {
                    "optimal_epoch": self.MSE_cv_idx_opt + 1,
                    "current_epoch": ti + 1,
                    "optimal_cv_mse": self.MSE_cv_epoch[self.MSE_cv_idx_opt],
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
            corr = np.zeros(self.ssModel.m)
            for i in range(self.ssModel.m):
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

        self.Plot = Plot(self.folder_name, self.modelName)

        self.Plot.NNPlot_epochs(
            self.N_Epochs,
            MSE_KF_dB_avg,
            self.MSE_test_dB_avg.cpu(),
            self.MSE_cv_dB_epoch.cpu(),
            self.MSE_train_dB_epoch.cpu(),
        )

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)
