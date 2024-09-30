'''
Model.py: This file contain different networks. 
'''
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *


class Network1(nn.Module):
    def __init__(self):
        super().__init__()

        # input 28 x 28 x 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.fc1 = nn.Linear(in_features=4 * 4 * 256, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, t):
        x = t
        # Block 1
        x = F.relu(self.conv1(x))      # (Out) 28 -> 26 | (RF) 1 -> 3 | (J) 1 -> 1
        x = F.relu(self.conv2(x))      # (Out) 26 -> 24 | (RF) 3 -> 5 | (J) 1 -> 1
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (Out) 24 -> 12 | (RF) 3 -> 4 | (J) 1 -> 2

        # Block 2
        x = F.relu(self.conv3(x))   # (Out) 12 -> 10 | (RF) 4 -> 8  | (J) 2 -> 2
        x = F.relu(self.conv4(x))   # (Out) 10 -> 8  | (RF) 8 -> 12 | (J) 2 -> 2
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (Out) 8 -> 4 | (RF) 12 -> 14 | (J) 2 -> 4

        x = x.reshape(-1, 4 * 4 * 256)
        # Block 3
        x = F.relu(self.fc1(x))     # output ->  5 x 5 x 64
        x = self.fc2(x)
        x = F.softmax(x, dim=1)     # output ->  1 x 10
        return x

