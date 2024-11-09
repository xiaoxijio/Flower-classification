import cv2
import numpy as np
import gzip
import os
from torchvision import models
from torch import nn

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def sort_contours(cnts, method):
    reverse = False
    i = 0

    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True

    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 外接矩阵
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))  # 打包排序

    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)  # 每个点的坐标和 x + y
    rect[0] = pts[np.argmin(s)]  # 返回和最小的点，即左上角的点
    rect[2] = pts[np.argmax(s)]  # 返回和最大的点，即右下角的点

    diff = np.diff(pts, axis=1)  # 每个点的坐标差值 y - x
    rect[1] = pts[np.argmin(diff)]  # 差值最小的点，即右上角的点
    rect[3] = pts[np.argmax(diff)]  # 差值最大的点，即左下角的点

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (top_left, top_right, bottom_right, bottom_left) = rect  # 找到四个坐标点(左上, 右上, 右下, 左下)

    widthA = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))  # 下边长
    widthB = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))  # 上边长
    maxWidth = max(int(widthA), int(widthB))  # 取最大边长( 宽 )

    heightA = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))  # 右边长
    heightB = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))  # 左边长
    maxHeight = max(int(heightA), int(heightB))  # 取最大边长( 高 )

    dst = np.array([  # 变换后的新坐标
        [0, 0],  # 左上
        [maxWidth - 1, 0],  # 右上
        [maxWidth - 1, maxHeight - 1],  # 右下
        [0, maxHeight - 1]], dtype="float32")  # 左下

    M = cv2.getPerspectiveTransform(rect, dst)  # 透视变换矩阵(将不规则的四边形映射到3*3矩形中)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # 将四边形在矩阵中 "拉正" 为矩形
    # 可以理解成一个不规则的四边形放到一个三维空间里，然后在三维空间里给它拉正为规则的四边形(不严谨！！！只是帮助理解！！！)

    return warped


def parse_mnist(minst_file_addr, flatten=False, one_hot=False):
    """解析MNIST二进制文件, 并返回解析结果
    输入参数:
        minst_file: MNIST数据集的文件地址. 类型: 字符串.
        flatten: bool, 默认Fasle. 是否将图片展开, 即(n张, 28, 28)变成(n张, 784)
        one_hot: bool, 默认Fasle. 标签是否采用one hot形式.

    返回值:
        解析后的numpy数组
    """
    minst_file_name = os.path.basename(minst_file_addr)  # 根据地址获取MNIST文件名字
    with gzip.open(filename=minst_file_addr, mode="rb") as minst_file:
        mnist_file_content = minst_file.read()
    if "label" in minst_file_name:  # 传入的为标签二进制编码文件地址
        data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8,
                             offset=8)  # MNIST标签文件的前8个字节为描述性内容，直接从第九个字节开始读取标签，并解析
        if one_hot:
            data_zeros = np.zeros(shape=(data.size, 10))
            for idx, label in enumerate(data):
                data_zeros[idx, label] = 1
            data = data_zeros
    else:  # 传入的为图片二进制编码文件地址
        data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8,
                             offset=16)  # MNIST图片文件的前16个字节为描述性内容，直接从第九个字节开始读取标签，并解析
        data = data.reshape(-1, 784) if flatten else data.reshape(-1, 28, 28)

    return data


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
