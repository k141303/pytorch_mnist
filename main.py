import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from trainer import MNISTTrainer
from model import Net

#画像を前処理してくれるモジュール
transform = transforms.Compose(
    [transforms.ToTensor(), #テンソルに変換
     transforms.Normalize((0.5, ), (0.5, ))])   #画像の各値を-1〜1に変換

#学習データ読み込み
train_set = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

#テストデータ読み込み
test_set = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)

print("TRAIN Length:",len(train_set))
print("TEST  Length:",len(test_set))
print("Data Shape:",train_set[0][0].size())

#バッチとかシャッフルとか色々設定してイテレータを作ってくれる
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False)

#モデル作成
net = Net(hidden =  256)

#trainerの作成
mnist = MNISTTrainer(net,train_dataloader,test_dataloader)

#学習の開始
epochs = 10
for epoch in range(epochs):
    mnist.train(epoch)  #学習
    mnist.test(epoch)   #テスト
