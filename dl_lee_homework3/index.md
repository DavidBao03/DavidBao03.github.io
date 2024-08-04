# 李宏毅机器学习 2023 Homework3


## 作业简介
利用CNN对食物的图片进行分类，一共有11个不同的类别

一些标准：

![hw3-1](/img/hw3-1.png)



## 调参记录
### 【Simple baseline】
老样子，跑通示例代码就行：

![hw3-2](/img/hw3-2.png)



### 【Medium baseline】
根据提示，我们需要先做一些图像增广，这里顺便把Report1在这里记录下来：
```python
homework_tfm = transforms.Compose([transforms.RandomGrayscale(), 
                transforms.RandomResizedCrop(128,(0.1, 1),(0.5, 2)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.3),
                transforms.GaussianBlur(7)])
init = transforms.Resize((128, 128))
img = Image.open('/kaggle/input/ml2023spring-hw3/train/0_0.jpg')
img = init(img)
display(img)
for _ in range(5):
    display(homework_tfm(img))
```
**效果如下**：

![hw3-3](/img/hw3-3.png)



>说实话，这变换之后我看着都费劲，不知道机器真的能看懂吗。。。

跑了70多个epoch之后，**没有过线**：

![hw3-4](/img/hw3-4.png)



感觉是自己的图像变换有问题，在网上找了一些资料（<https://zhuanlan.zhihu.com/p/430563265>）后，选择了以下的方案：
```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
 ])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
 ])
```
**结果有非常显著的提升**：

![hw3-5](/img/hw3-5.png)



### 【Strong Baseline】
根据提示，选一个定义好的模型来训练，我这里使用的是ResNet18。效果有，**但不多**：

![hw3-6](/img/hw3-6.png)

然后我继续尝试了ResNet34以及ResNet50，**但是也只有微弱的提升**：

![hw3-7](/img/hw3-7.png):



>小插曲：在最初定义模型的时候，由于官方文档没写需要给定num_classes参数，所以我直接忽略掉了这一项，但是没想到num_classes参数的默认值是1000！也就是意味着我上图跑的模型都是以1000类为目标的。在发现这点后，我立马去改了模型的定义加上了参数，重新训练了，结果居然大差不差，但是也是接近Strong baseline了：
>
>![hw3-8](/img/hw3-8.png)
>
>

### 【Boss Baseline】
最戏剧性的一幕是，当我想继续在Strong baseline的基础上选择更好的模型时，我选择了efficient net b3，但是**结果却直接过了Boss baseline**：

![hw3-9](/img/hw3-9.png)

于是我翻阅了efficient net的原始论文，使用了更强大的b4模型继续实验：

![hw3-10](/img/hw3-10.png)

在private上也获得了提升。

>在真正强大的模型面前，所有的cross validation和TTA这些技巧都显得微不足道啊。。

于是就这样稀里糊涂的过了Boss baseline，直接去下一个任务了^^

