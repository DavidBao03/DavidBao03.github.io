# 李宏毅机器学习 2023 Homework1


# 李宏毅机器学习 Hw1
>最开始是4 5个月前刚入门深度学习的时候做过一次，但是当时感觉只是囫囵吞枣般的做完了，没有认真的思考过里面的细节。最近在做datawhale夏令营时，被模型创建和训练伤透，深感自己的调参技术相当垃圾，于是重新捡起李老的作业从头认真的做一遍。

## 作业简介
一个回归问题，由若干不同症状的患者，根据他们的症状给出 covid-19 阳性的概率

一些标准:

<img src="/img/hw1-1.png" width="1000" height="250" />

## 调参记录

### 【Simple Baseline】
**只需要将原始代码跑一下就好了**

![hw1-2](/img/hw1-2.png)

### 【Medium + Strong Baseline】
修改特征选择（选择默认的除前35个之外的，即不选择地区作为学习的特征）：
```python
def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = list(range(35, raw_x_train.shape[1])) # TODO: Select suitable feature columns.
        
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid
```
再将select_all置为False：
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 5000,     # Number of epochs.            
    'batch_size': 256, 
    'learning_rate': 1e-5,              
    'early_stop': 600,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
```

运行出来的结果:
![hw1-3](/img/hw1-3.png)

>奇怪的是，助教给出的hint说通过选择特定的特征可以达到medium baseline，而要达到strong baseline还需要改进模型。但是仅仅通过选择特征就可以通过strong baseline了。看来所谓“数据远远大于模型”不无道理。

### 【Boss Baseline】
根据上一个baseline的经验，特征选择非常重要，因此我们选择调库来选择最好的k个特征：
```python
from sklearn.feature_selection import SelectKBest, f_regression

def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # TODO: Select suitable feature columns.
        selector = SelectKBest(score_func=f_regression, k=24)
        result = selector.fit(raw_x_train, y_train)
        idx = np.argsort(result.scores_)[::-1]
        feat_idx = list(np.sort(idx[:24]))
        
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid
```
>关于这段代码的解释（我的理解）：
>SeclectKBest是scikit中的一个函数，用于选择K个最好的特征，选择的标准则是由函数f_regression给出的，顾名思义，这是一个用于回归任务选择特征的函数。
>定义好selector之后（selector = SelectKBest(score_func=f_regression, k=24)，调用fit方法可以计算出所有特征的分数（result = selector.fit(raw_x_train, y_train)
>紧接着按照从大到小的顺序排列，选出最前面的k个下标组成切片，提取特征

**在不断的尝试下，k取20-24的时候效果最佳。**

接着是调整模型的大小和参数，加入了LeakyReLU和BatchNorm，以及Dropout：
```python
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
```

在训练方面，新增加学习率调整器，并且改用Adam：
```python
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'] * 10, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2,T_mult=2,eta_min=config['learning_rate'])
```
>关于CosineAnnealingWarmRestarts: 意思是可以将学习率从初始值，在2， 4， 8... 个（T_0*（n - 1） *  T_mult ）ephoch之间，逐渐下降到eta_min。一直重复这个周期

>此外，这里用AdamW的效果不如Adam，这里面可能有些东西没搞懂

最后是一些参数设置：
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 10000,     # Number of epochs.            
    'batch_size': 256, 
    'learning_rate': 1e-3,              
    'early_stop': 1000,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
```

**最后的结果：**
![hw1-4](/img/hw1-4.png)

只差一点就可以到boss baseline了，可能是特征的选择上还是没有做好。不过我也没有继续做了。
