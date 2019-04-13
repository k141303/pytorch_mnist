"""
深層学習モデルの定義
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn):
    def __init__(self,hidden):
        super(Net, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(hidden,hidden)
        self.fc2 = nn.Linear(hidden,10)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
