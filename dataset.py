import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    data:[103,13,13]
    label:int
    """

    def __init__(self, dataset_dir, labels_dir):
        self.dataset = torch.tensor(np.load(dataset_dir))
        self.labels = torch.tensor(np.load(labels_dir)).type(torch.long)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx, ...].unsqueeze(0), self.labels[idx, ...]
