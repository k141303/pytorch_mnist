"""
深層学習モデルの定義
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,hidden = 256):
        super(Net, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(28*28,hidden)
        self.fc2 = nn.Linear(hidden,10)

    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
