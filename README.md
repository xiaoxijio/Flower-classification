# 图像识别-102 花卉分类

### 花卉数据集

因为数据集太多了，上传不了，所以大家可以自己去[kaggle](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset)下载（下载挺快的，三百多M）

这是下载后包含的数据

![image](https://github.com/user-attachments/assets/43ffe2d9-4746-41ef-9431-49137380c14c)


我是放在'data/flower'目录下，大家如果不放在这个目录下更改一下我代码里文件位置就行

### 那么好！话归正题，如何白嫖别人训练好的模型

现在神经网络发展那么快，我们还自己从零开始敲摸着怎么构建神经网络那就大清亡了！垂涎欲滴地看着别人训练好的模型，不如直接零帧起手，顺手（红色警告！做合法牛马）。小小心思，[Pytorch](https://pytorch.org/vision/stable/models.html)早已看穿，为了促进人类文明的发展，与其..不如赠人玫瑰手留余香。目前已有如下这些零元购

![image](https://github.com/user-attachments/assets/14e923be-c117-42b6-90dc-e69316a5426e)


### 那么好！开始我们的模型调教！

什么导入数据，数据增强我就不说了，代码里注释的很清楚，主要说一下如何调教

比如我想用resnet网络，我们去官网一看，官网提供了如下几个

![image](https://github.com/user-attachments/assets/f67521c4-d73f-4833-8791-dc0e648e8e87)


那我们肯定不选对的，只选贵的。resnet152，看着就比别人牛逼，就是它了

1、调用前需要初始化一下 

```
if model_name == 'resnet':  
    model_ft = models.resnet152(pretrained=use_pretrained)  # use_pretrained下载人家的训练模型  
    set_parameter_requires_grad(model_ft, feature_extract)  # 因为我们用的别人训练好的参数, 所以冻结模型所有参数的梯度更新  
    num_ftrs = model_ft.fc.in_features  # 替换全连接层  因为人家最后的输出是 1000, 而我们是 102    model_ft.fc = nn.Sequential(  
        nn.Linear(num_ftrs, 102),  # 只有在使用 NLLLoss 时, 才需要在模型输出添加 LogSoftmax        
        nn.LogSoftmax(dim=1)  # 否则, 一般只需添加 Linear 层, 让损失函数自行处理  
    )  
    input_size = 224  # 模型的默认的输入尺寸是 224x224
```

比如我们的分类任务时102种花卉，而别人的模型输出是1000，所以我们首先要将别人模型的输出层size改成我们自己的size。

当然，我们都用别人的东西，也要适配别人，不能纯调教，那太丧心病狂了。比如resnet的架构在ImageNet上预训练时，输入尺寸就是224×224，那我们最好也将图片尺寸调整为224×224（如果你偏不，也不是不行）

2、冻结模型的特征提取层，只训练最后一层

在我上面代码里有一行

```
set_parameter_requires_grad(model_ft, feature_extract) # 因为我们用的别人训练好的参数, 所以冻结模型所有参数的梯度更新
```

能看这篇内容的，大都是发愤图强的学生，学生哪有那么好的gpu去训练模型。像现在这些牛逼哄哄的神经网络，人家顶级机器训练都是按天计算的，更何况...呜呜呜

既然我们要白嫖，那就把白嫖精神坚持到底！

```
def set_parameter_requires_grad(model, feature_extracting):  
    """ 冻结模型的特征提取层，只训练全连接层 """    
    if feature_extracting:  
        for param in model.parameters():  
            param.requires_grad = False
```

我们把前面那些乱七八糟的需要调整的参数全给冻起来，就用人家训练好的模型参数，给自己留个最后的全连接层训练就行啦，我们就已经很努力啦。然后把需要训练的全连接层参数给优化器，让它去慢慢优化。训练过程和验证过程我就不说了，人家要说的话都在酒里，我要说的话都在代码里。在训练过程中，将训练效果比较好的数据给保存下来哦，方便我们下次训练的时候不用从零开始了。

看看训练效果    

![train_1](https://github.com/user-attachments/assets/7d5830b9-68b6-483f-b311-4e6e43f9c84f)

![train_20](https://github.com/user-attachments/assets/432d782e-6a67-442d-9a71-1ba3d74851b3)

我就跑了个20轮，没多跑，大家电脑牛逼的可以多跑几轮。大概跑了个二十几分钟吧，准确率从32% --> 73%

效果远远不够啊，果然知识还得自己学才进脑子啊。但是不急，重新做人还来得及！

3、 加载预训练好的模型，重新做人

我们将之前训练效果好的模型数据加载出来，然后将之前投机取巧冻结的参数全部解冻，这一次，我要取回我的一切，电脑你要全力以赴啊！

在这一次训练中，我们需要将学习率调的再低一点，让模型在一个较优的情况下慢慢探索，不要一个步子迈大了，跌了个狗啃泥

我们需要调整的代码如下

```
feature_extract = False  # 不冻结
load_model = True  # 加载模型
optimizer_ft = optim.Adam(params_to_update, lr=1e-4)  # 调小学习率
```

看看效果

![test](https://github.com/user-attachments/assets/8907edcf-09d2-4c30-9cea-cfd975969f08)


这次我就跑了15轮，花费了大概半小时，效果一下提到了91%

好，我们已经学会如何使用别人的模型，那么接下了你要去攻打... 没错，学会了1+1，你就要会巴拉巴拉（此处脑补非常复杂难懂反人类的数学问题）

话说读书人的 怎么能叫 这叫迁移学习！（迁移学习为我正名）
