# 李宏毅机器学习 2023 Homework2


## 作业简介
给定若干段录音，将它分解成不同的小段（frames），通过深度学习的方法来确定这一段录音中讲话人说的是哪一个字（音素）。总而言之，这是一个分类问题。

一些标准：

![hw2-1](/img/hw2-1.png)

## 调参记录

### 【Simple Baseline】
**老样子，还是只需要把助教给的代码跑一遍就好“

![hw2-2](/img/hw2-2.png)

### 【Medium Baseline】
根据提示，达到medium的条件是将合适的多个frames拼接在一起，这样可以最大限度的保留整个音素的信息。此外，还需要在模型中增加更多的层。

首先在block中添加更多的层，并且使用dropout和batchnorm
```python
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.block(x)
        return x
```

接着改变隐藏层的大小，连接更多的frames（这里n取11）：
```python
# data prarameters
concat_nframes = 11              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.8               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 1213                        # random seed
batch_size = 512                # batch size
num_epoch = 10                   # the number of training epoch
learning_rate = 1e-4         # learning rate
model_path = './model.ckpt'     # the path where the checkpoint will be saved

# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 2               # the number of hidden layers
hidden_dim = 512                # the hidden dim
```

**运行结果：**
![hw2-3](/img/hw2-3.png)

可以看到，效果并不理想。于是转而使用更深的网络，更宽的层（layers=6，dim=1024），并且连接更多的frames（n=17），**结果更好了：**
![hw2-4](/img/hw2-4.png)

>思考题：课件中让我们做一个小实验，即更深，更窄的层好 还是 更浅，更宽的层好。照着课件上的思路，再根据上面的模型重新跑了一遍，这次layers=2， dim=1750，结果明显好于上面的模型：
>![hw2-5](/img/hw2-5.png)
>但是根据CNN的思想来说，应该是更深的层效果会更好才对，不知道这里我是不是实验有误，或许应该再多跑几次才能得出结论的。

### 【Strong Baseline】
我们根据上面的思路，先进一步加深模型，取layers=12， dim=1024。此外，在模型输出最后一层加上softMax，但是**结果却出奇的差：**
![hw2-6](/img/hw2-6.png)
于是想到会不会是softMax的问题，于是去掉后重新做实验，**发现效果变好了**，证明确实是softmax的问题：
![hw2-7](/img/hw2-7.png)

>在查阅资料之后（https://www.zhihu.com/question/456082205/answer/1869602054）
>发现其实crossentry的损失函数是默认加了一层softMax的，所以如果在模型中再加一层的话会导致模型难以收敛。

### 【Boss Baseline】
助教提示的slides里写道，如果需要过boss baseline的话，需要用到RNN。我这里首先想到的是用LSTM。根据我之前看到的一篇文章（https://www.zhihu.com/question/25097993/answer/3410497378） 并按照这个顺序来从头构建这个模型，正好实践一下。
根据**第一条建议**，我构建出了以下模型：
```python
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=hidden_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
```
其中layers=2，dim=1024，并且成功在[10000, 2000]的数据集上过拟合：
![hw2-8](/img/hw2-8.png)

>这里我还尝试了layers=2，3， dim=512，1024， 2048的所有排列组合，最后发现这有选择的这个组合下train_loss的曲线最像log函数，跟建议所说一致。

再根据**第五，六条建议**，设定学习率为1e-4，使用Adam和CosineAnnealingLR。
再根据**第七条建议**，使用梯度裁剪...

>参考的太多了！自己看博客吧... 

跑了一晚上之后，**结果不尽人意：**
![hw2-9](/img/hw2-9.png)

之后在网上参考了大量的博客和文章，最后把模型继续加深：
```python
import torch.nn as nn
import torch.nn.init as init

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        
        self.hidden_layers = 5
        self.hidden_dim = 512
        self.input_dim = 39
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.hidden_layers,dropout=0.25,batch_first=True,bidirectional=True)
        self.norm = nn.LayerNorm(self.hidden_dim * 2)
        self.relu = nn.ReLU()

        self.fc = nn.Sequential(
            BasicBlock(self.hidden_dim * 2, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.dropout = nn.Dropout(0.25)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # input to hidden weights
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # hidden to hidden weights
                init.orthogonal_(param.data)
            elif 'bias' in name:  # biases
                init.zeros_(param.data)
            else:
                init.he_uniform_(param.data)

    def forward(self, x):
        x = x.view(x.shape[0], concat_nframes, 39)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
```

再根据**第二条建议**，在[60000, 3000] 和 [80000, 4000] 的小数据集上分别调参，最后确定了超参数:
```python
# data prarameters
concat_nframes = 81              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.95               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 1213                        # random seed
batch_size = 256                # batch size
num_epoch = 20                   # the number of training epoch
learning_rate = 2e-4         # learning rate
model_path = './model.ckpt'     # the path where the checkpoint will be saved

# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 4               # the number of hidden layers
hidden_dim = 1024                # the hidden dim

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2,eta_min=0.1 * learning_rate)
```
得到的loss曲线如下所示：
![hw2-10](/img/hw2-10.png)
可以看到，在这个参数下模型拟合的不错。

最终在跑了20ephoch（约用了16个小时），得到了**最后的结果**：
![hw2-11](/img/hw2-11.png)
可惜的是仍然没有过boss baseline。loss曲线如下：
![hw2-12](/img/hw2-12.png)
可以看到loss在最后并没有完全收敛（甚至train loss 还没有超过 val loss），于是决定再多跑几个epoch。
在进行多5轮的训练后，发现模型已经收敛了，再次提交效果**并没有得到提升**：
![hw2-13](/img/hw2-13.png)
感觉很可惜，毕竟只差一点点了。但是从头再训练一次花费的时间太多了，而且对学习没有太大的提升了，于是就先这样了吧！
