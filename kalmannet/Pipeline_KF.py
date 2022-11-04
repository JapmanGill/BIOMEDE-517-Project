import torch
import torch.nn as nn
import random
import time
import numpy as np

import wandb
from utils.utils import compute_correlation
from prettytable import PrettyTable
import optuna


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

    def set_model(self, model):
        self.model = model

    def set_training_params(self, n_epochs, learning_rate, weight_decay):
        self.n_epochs = n_epochs  # Number of Training Epochs
        self.learning_rate = learning_rate  # Learning Rate
        self.weight_decay = weight_decay  # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction="mean")

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def train(
        self,
        train_dataloader,
        cv_dataloader,
        compute_val_every=100,
        stop_at_iterations=500,
        trial=None,
    ):
        # optimal_mse = 1000
        # optimal_correlation = torch.empty([self.ssModel.m])
        # optimal_epoch = -1

        # self.count_parameters(self.model)
        for epoch in range(self.n_epochs):
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
                y = torch.index_select(y, 1, self.good_chans.to(self.model.device))
                x = loader_dict["states_hist"]
                x = torch.cat(
                    [x, torch.ones(x.shape[0], 1, x.shape[2]).to(self.model.device)], 1
                )
                x_0 = loader_dict["initial_states"]
                x_0 = torch.cat(
                    [x_0, torch.ones(x_0.shape[0], 1).to(self.model.device)], 1
                )
                # Initialize hidden state: necessary for backprop
                # FIXME: initialize one hidden state per sequence
                self.model.init_hidden()
                self.optimizer.zero_grad()
                # Initialize sequence for KalmanFilter
                # self.model.InitSequence(x_0[0, :])
                # Do forward pass of full batch
                x_hat = self.model.forward_batch(y, x_0).to(device=self.model.device)
                # Compute MSE loss
                if self.pred_type == "v":
                    loss = self.loss_fn(x_hat[:, 2:4, :], x[:, 2:4, :])
                    train_loss[i, :] = (
                        ((x_hat[:, 2:4, :] - x[:, 2:4, :]) ** 2)
                        .mean(axis=[0, 2])
                        .detach()
                    )
                else:
                    loss = self.loss_fn(x_hat[:, :4, :], x[:, :4, :])
                    train_loss[i, :] = (
                        ((x_hat[:, :4, :] - x[:, :4, :]) ** 2)
                        .mean(axis=[0, 2])
                        .detach()
                    )
                # Backpropagate and update weights
                loss.backward()

                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)

                self.optimizer.step()
                # Compute correlation between x and x_hat
                if self.pred_type == "v":
                    corr = compute_correlation(
                        x[:, 2:4, :].detach().cpu(), x_hat[:, 2:4, :].detach().cpu()
                    )
                else:
                    corr = compute_correlation(
                        x[:, :4, :].detach().cpu(), x_hat[:, :4, :].detach().cpu()
                    )

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
                    val_loss, val_corr = self.compute_validation_results(cv_dataloader)
                    log_dict.update(
                        {"val_loss": val_loss.mean(), "val_corr": val_corr.mean()}
                    )
                    print(
                        f"Validation loss: {val_loss.mean()}, corr: {val_corr.mean()}"
                    )
                    if trial is not None:
                        trial.report(val_loss.mean(), num_iteration)

                        if trial.should_prune():
                            wandb.run.summary["state"] = "pruned"
                            wandb.finish(quiet=True)
                            raise optuna.exceptions.TrialPruned()

                # Log to Wandb
                wandb.log(log_dict)

                if (
                    stop_at_iterations is not None
                    and num_iteration == stop_at_iterations
                ):
                    val_loss, val_corr = self.compute_validation_results(cv_dataloader)
                    wandb.run.summary["final val_loss"] = val_loss.mean()
                    wandb.run.summary["state"] = "completed"
                    wandb.finish(quiet=True)
                    return val_loss.mean()

            # mse = np.nanmean(val_loss.cpu())
            # print("Validation", mse, np.nanmean(val_corr.cpu(), 0).tolist())
            # if mse < optimal_mse:
            #     optimal_mse = mse
            #     optimal_correlation = np.nanmean(val_corr.cpu(), 0).tolist()
            #     optimal_epoch = epoch
            #     torch.save(self.model, self.modelFileName)
            #     print(f"Saving model with MSE {optimal_mse}")

    def compute_validation_results(self, cv_dataloader):
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
            y = torch.index_select(y, 1, self.good_chans.to(self.model.device))
            x = loader_dict["states"]
            x = torch.cat([x, torch.ones(x.shape[0], 1).to(self.model.device)], 1)
            x_0 = loader_dict["initial_states"]
            x_0 = torch.cat([x_0, torch.ones(x_0.shape[0], 1).to(self.model.device)], 1)

            # Initialize hidden state: necessary for backprop
            self.model.init_hidden()
            # Initialize sequence for KalmanFilter
            self.model.init_sequence(x_0[0, :])
            # Run model on validation set
            # Output is (seq_len, m)
            x_hat = self.model.forward_sequence(y.T).T.to(self.model.device)
            # Compute MSE loss and correlation
            if self.pred_type == "v":
                val_loss[j, :] = (
                    ((x_hat[:, 2:4] - x[:, 2:4]) ** 2).mean(axis=[0]).detach()
                )
                val_corr[j, :] = compute_correlation(
                    x[:, 2:4].detach().cpu().T,
                    x_hat[:, 2:4].detach().cpu().T,
                )
            else:
                val_loss[j, :] = (
                    ((x_hat[:, :4] - x[:, :4]) ** 2).mean(axis=[0]).detach()
                )
                val_corr[j, :] = compute_correlation(
                    x[:, :4].detach().cpu().T,
                    x_hat[:, :4].detach().cpu().T,
                )
        return val_loss, val_corr

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
