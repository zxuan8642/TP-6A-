"""
6A组 白盒对抗样本攻防实验 - 模型定义模块
包含：归一化层、适配CIFAR-10的ResNet-18模型
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18
from data_utils import cifar10_mean, cifar10_std, device

class Normalize(nn.Module):
    """内嵌归一化层，保证白盒梯度计算在原始像素空间进行"""
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std  = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class ResNet18_CIFAR(nn.Module):
    """适配CIFAR-10的ResNet-18，修改首层卷积、移除初始池化层"""
    def __init__(self):
        super().__init__()
        self.normalize = Normalize(cifar10_mean, cifar10_std)
        self.backbone = resnet18(weights=None)

        # 适配32×32图像输入
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        # 输出层适配10分类
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 10)

    def forward(self, x, return_features=False):
        x = self.normalize(x)
        features = None
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        out = self.backbone.fc(features)

        if return_features:
            return out, features
        return out