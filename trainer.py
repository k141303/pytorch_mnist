import torch
from torch.utils.data import DataLoader
from model import Net
from tqdm import tqdm

class MNISTTrainer:
    def __init__(self, net : Net, train_data : DataLoader, test_data : DataLoader, lr: float = 0.01, use_gpu = False):
        self.train_data = train_data
        self.test_data = test_data
        self.net = net
        self.criterion = nn.CrossEntropyLoss()   #誤差関数
        self.optim = torch.optim.SGD(net.parameters(), lr=lr)   #最適化手法にSGDを適用
        self.device = torch.device("cuda:0" if use_gpu else "cpu")  #デバイス選択

        self.net.to(self.device)    #デバイスに送る

        if torch.cuda.device_count() > 1 and use_gpu:   #複数gpuを使用できる場合の処理
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

    def train(self):   #学習
        return iteration(self.train_data,train = True)

    def test(self):    #テスト
        return iteration(self.test_data,train = False)

    def iteration(self,data,train = False):
        data_iter = tqdm.tqdm(enumerate(data), desc="EP:%d" % (epoch), total=len(data_iter), bar_format="{l_bar}{r_bar}")
        sum_loss,sum_correct,sum_element = 0,0,0

        for i,(img,label) in data_iter:
            t = self.net(img)   #forward計算
            loss = self.criterion(t, label)    #誤差を計算

            if train:   #学習時
                self.optim.zero_grad()  #勾配初期化
                loss.backward() #誤差逆伝搬
                self.optim.step()   #モデルパラメータの更新

            sum_loss += loss.item() #lossの合計
            sum_correct += t.argmax(dim=-1).eq(label).sum()  #正答率計算
            sum_element += label.nelement() #総画像数の合計

        print("loss:",sum_loss/len(data))
        print("correct",sum_correct/sum_element*100.0)
