import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from Utils import *

'''
This file contain all the Networks: 
'''


'''
This is First network created for S5
'''
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

'''
Network2 : This model is for Mnist dataset.
Not used 
'''
class Network2(nn.Module):
    def __init__(self):
        super().__init__()

        # input 28 x 28 x 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(in_features=4 * 4 * 256, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, t):
        x = t
        # Block 1
        x = F.relu(self.conv1(x))      # (Out) 28 -> 26 | (RF) 1 -> 3 | (J) 1 -> 1
        x = F.relu(self.conv2(x))      # (Out) 26 -> 24 | (RF) 3 -> 5 | (J) 1 -> 1
        x = self.pool1(x)              # (Out) 24 -> 12 | (RF) 3 -> 4 | (J) 1 -> 2
        x = self.dropout(x)

        # Block 2
        x = F.relu(self.conv3(x))   # (Out) 12 -> 10 | (RF) 4 -> 8  | (J) 2 -> 2
        x = F.relu(self.conv4(x))   # (Out) 10 -> 8  | (RF) 8 -> 12 | (J) 2 -> 2
        x = self.pool2(x)           # (Out) 8 -> 4 | (RF) 12 -> 14 | (J) 2 -> 4
        x = self.dropout(x)

        x = x.reshape(-1, 4 * 4 * 256)
        # Block 3
        x = F.relu(self.fc1(x))     # output ->  5 x 5 x 64
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)     # output ->  1 x 10
        return x

'''
Network 3:
Expected 99.4% validation accuracy
Less than 20k Parameters
Less than 20 Epochs
Have used BN, Dropout,
(Optional): a Fully connected layer, have used GAP.
'''
class Network3(nn.Module):
    def __init__(self):
        super().__init__()

        # input 28 x 28 x 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Linear(in_features=1 * 1 * 32, out_features=10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, t):
        x = t
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))      # (Out) 28 -> 26 | (RF) 1 -> 3 | (J) 1 -> 1
        x = self.pool1(x)                        # (Out) 24 -> 12 | (RF) 3 -> 4 | (J) 1 -> 2
        x = self.dropout(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))      # (Out) 12 -> 10 | (RF) 4 -> 8 | (J) 2 -> 2
        x = self.pool1(x)                        # (Out) 10 -> 5  | (RF) 8 -> 12 | (J) 2 -> 4
        x = self.dropout(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))     # (Out) 5 -> 3 | (RF) 12 -> 20  | (J) 4 -> 4
        x = self.dropout(x)

        x = self.gap(x) 
        x = x.reshape(x.size(0), -1) # Flatten the tensor
        # Block 3
        x = self.fc(x)              # output ->  1 x 1 x 32
        x = F.softmax(x, dim=1)     # output ->  1 x 10
        return x

