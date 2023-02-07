import pandas as pd
import os

import torch
from torch.utils.data import Dataset
import numpy as np


class MNISTdataset(Dataset):
    """
    Dataset for mnist
    """
    data_dir = "../data/mnist"

    def __init__(self, isTrain, transform_x=None, transform_y=None, data_dir=None):
        np.random.seed(0)

        if data_dir is not None:
            self.data_dir = data_dir

        train_file = os.path.join(self.data_dir, "train.csv")
        test_file = os.path.join(self.data_dir, "test.csv")

        self.isTrain = isTrain

        if self.isTrain:
            df = pd.read_csv(train_file)
            self.y, self.x = df.iloc[:,0].to_numpy(), df.iloc[:,1:].to_numpy()
        else:
            df = pd.read_csv(test_file)
            self.x = df.to_numpy()

        self.transform_x = transform_x
        self.transform_y = transform_y
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if not self.isTrain:
            x = self.x[idx].reshape(-1, 28, 28)
            if self.transform_x:
                return self.transform_x(x)
            return x

        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, y = self.x[idx].reshape(-1, 28, 28), self.y[idx]

        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)
        
        return x, y