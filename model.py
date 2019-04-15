"""
深層学習モデルの定義
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,hidden = 784):
        super(Net, self).__init__()
        self.hidden = hidden    #隠れ層のサイズ
        self.fc1 = nn.Linear(28*28,hidden)  #入力層
        self.fc2 = nn.Linear(hidden,10) #出力層

    def forward(self,x):
        x = x.view(-1,28*28)    #入力を(バッチサイズ,1,28,28)から(バッチサイズ,28*28)に変換する
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
