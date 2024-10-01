# Prepare the dataset class
# DATASET : class name
# By : Dhruv Vyas on 01-09-2024

import numpy as np
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import torch


class MnistDataSet(Dataset):
    def __init__(self, batchsize=10):
        super().__init__()
        self.batch_size = batchsize
        # Train data transformations
        self.train_transforms = transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15., 15.), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        # Test data transformations
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_set = torchvision.datasets.MNIST(train=True,
                                                           root='./data',
                                                           download=True,
                                                           transform=self.train_transforms)
        self.test_set = torchvision.datasets.MNIST(train=False,
                                                          root='./data',
                                                          download=True,
                                                          transform=self.test_transforms)

    def get_train_loader(self, kwargs):
        return torch.utils.data.DataLoader(self.train_set, **kwargs)

    def get_test_loader(self, kwargs):
        return torch.utils.data.DataLoader(self.test_set, **kwargs)

    def __getitem__(self, index):
        label, image = self.train_set[index]
        return label, image

    def __len__(self):
        return len(self.train_set)
