from torch.utils.data import Dataset
import torch


class FingerFlexionDataset(Dataset):
    def __init__(
        self, X, Y, X_0, x_mean=None, x_std=None, y_mean=None, y_std=None
    ) -> None:
        """
        X(num_examples, 4, seq_length): state is (pos_index, pos_mrs, vel_index, vel_mrs)
        Y(num_examples, channels, seq_length)
        X_0(num_examples, 4)
        """
        self.X = X
        self.Y = Y
        self.X_0 = X_0
        self.normalize = True
        if None in [x_mean, x_std, y_mean, y_std]:
            print("Won't normalize")
            self.normalize = False
        else:
            self.x_mean = x_mean
            self.x_std = x_std
            self.y_mean = y_mean
            self.y_std = y_std
            self.normalize = True

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, index):
        x = self.X[index, :, :]
        y = self.Y[index, :, :]
        x_0 = self.X_0[index, :]
        if self.normalize:
            # x = torch.div(x - self.x_mean[0, :, None], self.x_std[0, :, None])
            y = torch.div(y - self.y_mean[0, :, None], self.y_std[0, :, None])
            # x_0 = torch.div(x_0 - self.x_mean[0, :, None], self.x_std[0, :, None])
            pass
        return y, x, x_0
