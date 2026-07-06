import torch
import torch.nn as nn
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std  = (0.2023, 0.1994, 0.2010)

class Normalize(nn.Module):
    """内嵌归一化层，保证白盒梯度计算在原始像素空间进行"""
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std  = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class ResNet18_CIFAR(nn.Module):
    """适合CIFAR-10的ResNet-18（修改首层卷积，移除第一个池化）"""
    def __init__(self):
        super().__init__()
        self.normalize = Normalize(cifar10_mean, cifar10_std)
        self.backbone = resnet18(weights=None)
        # 适配32x32输入
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
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