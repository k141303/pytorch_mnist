import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

#画像を簡単に扱うためのモジュール
transform = transforms.Compose(
    [transforms.ToTensor(), #テンソルに変換
     transforms.Normalize((0.5, ), (0.5, ))])   #画像の各値を-1〜1に変換

#学習データ読み込み
trainset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

#テストデータ読み込み
testset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)

print("TRAIN Length:",len(trainset))
print("TEST  Length:",len(testset))
print("Data Shape:",trainset[0][0].size())
