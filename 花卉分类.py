import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from unit import initialize_model
import time
import json

data_tranforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),  # 在(-45,45)角度随机旋转 旋转空白处填充黑色 0
        transforms.CenterCrop(224),  # 中心裁剪 (旋转后图片size变大, 给ta剪掉!)
        transforms.RandomHorizontalFlip(p=0.5),  # 概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 概率垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 概率调整亮度、对比度、饱和度和色调
        transforms.RandomGrayscale(p=0.025),  # 概率灰度
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化让数据分布接近标准正态分布
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'data/flower/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
filename = 'model/checkpoint.pth'  # 把训练好的模型保存到这
batch_size = 8  # 如果你的电脑过硬 调大它！ 16！ 32！ 64！ 128！！！(不知道在燃什么)

# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html?highlight=imagefolder
# ImageFolder自动为文件夹中的每个类别分配标签, 不懂的看上面链接
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_tranforms[x]) for x in ['train', 'valid']}
# 每次从 dataloaders 中提取一个样本时，transforms 都会在原图上随机应用一系列变换
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}  # 数据集大小
class_names = image_datasets['train'].classes  # 标签
with open('data/flower/cat_to_name.json', 'r') as f:  # 读取标签对应的花卉名字
    cat_to_name = json.load(f)


def train(model, data, criterion, opt):
    print('训练开始')
    start = time.time()  # 可以看看训练用了多长时间
    running_loss = 0.0
    correct = 0

    model.train()
    for inputs, labels in data['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()
        predicted = torch.max(outputs, 1)[1]
        correct += torch.sum(predicted == labels.data)  # 求和预测对的
        running_loss += loss.item() * inputs.size(0)  # 累加损失

    accuracy = correct / len(data['train'].dataset)
    epoch_time = time.time() - start
    print('train_loss:', running_loss / len(dataloaders['train'].dataset))
    print(f"训练准确率: {accuracy * 100:.2f}%")  # 看一下训练过程准确率和验证集上准确率, 如果相差过大就过拟合了
    print('训练时长 {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
    print('训练完成')


@torch.no_grad()
def valid(model, data, criterion):
    print('验证开始')
    correct = 0

    model.eval()
    for inputs, labels in data['valid']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        predicted = torch.max(outputs, 1)[1]
        correct += torch.sum(predicted == labels.data)  # 求和预测对的

    accuracy = correct / len(data['valid'].dataset)
    print(f"验证准确率: {accuracy * 100:.2f}%")
    print('验证结束')
    return accuracy


if __name__ == "__main__":
    feature_extract = False  # 是否用别人训练好的模型参数 (一开始训练的时候用, 等自己训练几回合之后 False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # gpu还是cpu 没有gpu就嗝屁了

    # https://pytorch.org/vision/stable/models.html  具体有啥模型可以去官网看
    model_name = 'resnet'  # 大家也可以选择其他的模型, 比如牛逼哄哄 EfficientNet 垂垂老矣 AlexNet 电脑不行的用 MobileNet
    model_ft, input_size = initialize_model(model_name, 102, feature_extract)
    model_ft = model_ft.to(device)
    # print(model_ft)  # 这里可以看一下模型结构

    params_to_update = model_ft.parameters()  # 模型中需要更新的参数
    if feature_extract:  # 如果我们用人家的参数, 那么我们需要更新的参数只有全连接层, 重写 params_to_update
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print(name)  # 这里看一下 可以更新的参数是不是只有最后全连接层的 w和 b
    optimizer_ft = optim.Adam(params_to_update, lr=1e-4)  # 学习率衰减也可以直接加个 weight_decay 参数
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 学习率每7个epoch衰减为原来的1/10
    criterion = nn.NLLLoss()  # 如果模型后面没改成LogSoftmax, 可以直接用 CrossEntropyLoss

    load_model = True  # 是否加载模型(一开始我们还没有训练好的模型, 后面有了再开)
    best_acc = 0  # 保存最好的准确率
    if load_model and os.path.exists(filename):
        checkpoint = torch.load(filename)
        model_ft.load_state_dict(checkpoint['state_dict'])
        # optimizer_ft.load_state_dict(checkpoint['optimizer'])  # 如果解冻，注释这行
        best_acc = checkpoint['best_acc']

    for epoch in range(20):  # 电脑牛逼的多上几轮
        train(model_ft, dataloaders, criterion, optimizer_ft)  # train() 函数中进行的更新会直接反映在 model_ft
        epoch_acc = valid(model_ft, dataloaders, criterion)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            state = {  # state_dict变量存放训练过程中需要学习的权重和偏执系数
                'state_dict': model_ft.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer_ft.state_dict(),
            }
            torch.save(state, filename)
