"""
深層学習モデルの定義
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    """
    フィードフォワードネットワーク(F(F)NN)
    """
    def __init__(self,hidden = 784):
        super(FNN, self).__init__()
        self.hidden = hidden    #隠れ層のサイズ
        self.fc1 = nn.Linear(28*28,hidden)  #入力層
        self.fc2 = nn.Linear(hidden,10) #出力層

    def forward(self,x):
        x = x.view(-1,28*28)    #入力を(バッチサイズ,1,28,28)から(バッチサイズ,28*28)に変換する
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    """
    畳み込みニューラルネットワーク(CNN)
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3,padding = 1) #(バッチサイズ,1,28,28) => (バッチサイズ,10,28,28)
        self.conv2 = nn.Conv2d(10, 5, 3) #(バッチサイズ,10,28,28) => (バッチサイズ,5,26,26)
        self.pool = nn.MaxPool2d(2, 2)  #(バッチサイズ,5,26,26) => (バッチサイズ,5,13,13)
        self.fc1 = nn.Linear(5*13*13, 10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1,5*13*13)  #(バッチサイズ,5,13,13) => (バッチサイズ,5*13*13)
        x = self.fc1(x)
        return x
