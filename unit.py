import numpy as np
import os
from torchvision import models
from torch import nn

def set_parameter_requires_grad(model, feature_extracting):
    """ 冻结模型的特征提取层，只训练全连接层 """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """ 初始化 '我们自己的' 模型 """
    model_ft = None
    input_size = 0
    if model_name == 'resnet':
        model_ft = models.resnet152(pretrained=use_pretrained)  # use_pretrained下载人家的训练模型
        set_parameter_requires_grad(model_ft, feature_extract)  # 因为我们用的别人训练好的参数, 所以冻结模型所有参数的梯度更新
        num_ftrs = model_ft.fc.in_features  # 替换全连接层  因为人家最后的输出是 1000, 而我们是 102
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),  # 只有在使用 NLLLoss 时, 才需要在模型输出添加 LogSoftmax
            nn.LogSoftmax(dim=1)  # 否则, 一般只需添加 Linear 层, 让损失函数自行处理
        )
        input_size = 224  # 模型的默认的输入尺寸是 224x224

    elif model_name == "efficientnet":
        model_ft = models.efficientnet_b0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features  # 获取最后的分类层的输入特征数
        model_ft.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),  # 替换全连接层，输出类别数
            nn.LogSoftmax(dim=1)  # 如果用 NLLLoss，可以添加 LogSoftmax 层
        )
        input_size = 224

    elif model_name == "mobilenet":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)  # 如果用 NLLLoss，可以添加 LogSoftmax 层
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_frts = model_ft.classifier[6].in_features  # 第 6 层为最终输出层
        model_ft.classifier[6] = nn.Linear(num_frts, num_classes)
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_frts = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_frts, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # SqueezeNet 的分类层使用 Conv2d 而非全连接层, 这里直接替换 classifier 输出通道数
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_frts = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_frts, num_classes)
        input_size = 224

    else:
        print("没有这模型")
        exit()

    return model_ft, input_size
