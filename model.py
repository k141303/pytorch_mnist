"""
深層学習モデルの定義
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn):
    def __init__(self,hidden = 256):
        super(Net, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(27*27,hidden)
        self.fc2 = nn.Linear(hidden,10)

    def forward(self,x):
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
