import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os

# ==================== 全局配置常量 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
batch_size = 128
num_epochs = 80
adv_epochs = 40
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
num_workers = 2

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std  = (0.2023, 0.1994, 0.2010)

# 对抗攻击默认参数
epsilon = 8/255
pgd_alpha = 2/255
pgd_steps = 10
deepfool_max_iter = 50
deepfool_overshoot = 0.02

# JPEG压缩防御质量参数
jpeg_quality = 75

# 固定随机种子
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 输出目录
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==================== 数据集加载工具 ====================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, pin_memory=True)
testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 干净模型准确率评估工具（公共函数）
def evaluate_clean(model, loader, device):
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