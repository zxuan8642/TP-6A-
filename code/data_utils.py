"""
6A组 白盒对抗样本攻防实验 - 数据与全局配置模块
包含：全局参数、数据集加载、通用评估函数、工具函数
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os

# ==================== 全局配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
batch_size = 128
num_epochs = 120
adv_epochs = 40
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
num_workers = 2

# CIFAR-10归一化参数
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std  = (0.2023, 0.1994, 0.2010)

# 对抗攻击默认参数
epsilon = 16/255
pgd_alpha = 2/255
pgd_steps = 20
deepfool_max_iter = 50
deepfool_overshoot = 0.02

# JPEG压缩防御质量参数
jpeg_quality = 75

# 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ==================== 工具函数 ====================
def set_seed(seed=seed):
    """固定全局随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dirs():
    """创建所有输出目录"""
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

# ==================== 数据集加载 ====================
def get_dataloaders():
    """加载CIFAR-10训练集与测试集，返回DataLoader"""
    # 训练集数据增强，测试集仅转张量，归一化内嵌至模型
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return trainloader, testloader

# ==================== 通用评估函数 ====================
def evaluate_clean(model, loader, device):
    """干净样本测试集准确率评估"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# 初始化
set_seed()
create_dirs()