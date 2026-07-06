"""
6A组 白盒对抗样本攻防实验 完整代码
CIFAR-10 + ResNet-18 + FGSM/PGD/DeepFool + 对抗训练 + JPEG压缩防御
包含：攻击效果评估、参数分析、军备竞赛验证、可视化与t-SNE分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from sklearn.manifold import TSNE
from PIL import Image
import io

# ==================== 全局配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
batch_size = 128
num_epochs = 80
adv_epochs = 40
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
num_workers = 2

# CIFAR-10归一化参数
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

# 创建输出目录
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==================== 模块1：数据与环境基座 ====================
# 训练集数据增强，测试集仅转张量，归一化内嵌至模型
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

# ==================== 模块2：目标模型基座 ====================
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
        from torchvision.models import resnet18
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

def train_standard(model, trainloader, epochs, device):
    """标准监督训练，保存测试集最优权重"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)
    best_acc = 0.0
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=running_loss/(pbar.n+1), acc=100.*correct/total)
        scheduler.step()
        acc = evaluate_clean(model, testloader, device)
        print(f'Epoch {epoch+1}: Test Accuracy = {acc:.2f}%')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/resnet18_clean.pth")
    print(f'Best Clean Accuracy: {best_acc:.2f}%')

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

# ==================== 模块3：白盒攻击算法集 ====================
def fgsm_attack(model, images, labels, epsilon, targeted=False):
    """
    FGSM快速梯度符号法
    非目标攻击：沿损失梯度增加方向添加L∞扰动
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad_sign = images.grad.sign()
    if targeted:
        adv_images = images - epsilon * grad_sign
    else:
        adv_images = images + epsilon * grad_sign
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images.detach()

def pgd_attack(model, images, labels, epsilon, alpha, steps, random_start=True):
    """
    PGD投影梯度下降迭代攻击
    每次迭代后将扰动约束至[-ε,ε]区间并裁剪像素值
    """
    adv_images = images.clone().detach()
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, 0, 1)

    for _ in range(steps):
        adv_images = adv_images.clone().detach().requires_grad_(True)
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            eta = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = torch.clamp(images + eta, 0, 1)
    return adv_images.detach()

def deepfool_attack(model, images, labels=None, max_iter=50, overshoot=0.02, device=device):
    """
    DeepFool最小范数扰动攻击
    迭代求解至样本跨越决策边界，添加过冲系数增强攻击效果
    """
    adv_images = images.clone().detach().to(device)
    batch_size = images.shape[0]
    model.eval()
    for idx in range(batch_size):
        x = adv_images[idx:idx+1].clone().detach().requires_grad_(True)
        with torch.no_grad():
            outputs = model(x)
            _, pred = outputs.max(1)
            pred = pred.item()
        iter_num = 0
        w = torch.zeros_like(x)
        r_tot = torch.zeros_like(x)
        current_pred = pred
        while current_pred == pred and iter_num < max_iter:
            x.requires_grad = True
            outputs = model(x)
            loss_original = outputs[0, pred]
            grad_original = torch.autograd.grad(loss_original, x, retain_graph=True, create_graph=False)[0]
            min_dist = float('inf')
            w_k = None
            for k in range(10):
                if k == pred:
                    continue
                loss_k = outputs[0, k]
                grad_k = torch.autograd.grad(loss_k, x, retain_graph=True, create_graph=False)[0]
                w_diff = grad_k - grad_original
                f_diff = loss_k - loss_original
                dist = torch.abs(f_diff) / (torch.norm(w_diff.flatten()) + 1e-8)
                if dist < min_dist:
                    min_dist = dist
                    w_k = w_diff
            if w_k is None:
                break
            ri = (min_dist + 1e-4) * w_k / (torch.norm(w_k.flatten()) + 1e-8)
            r_tot = r_tot + ri
            with torch.no_grad():
                x = (x + (1 + overshoot) * ri).detach()
                x = torch.clamp(x, 0, 1)
            iter_num += 1
            with torch.no_grad():
                new_out = model(x)
                _, new_pred = new_out.max(1)
                current_pred = new_pred.item()
        adv_images[idx] = x.detach().squeeze(0)
    return adv_images

def test_attack(model, loader, attack_fn, device, **attack_kwargs):
    """
    攻击效果评估
    返回攻击成功率ASR、对抗样本准确率、原始干净样本准确率
    """
    model.eval()
    correct_orig = 0
    correct_adv = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            orig_out = model(imgs)
            _, orig_pred = orig_out.max(1)
            correct_orig += orig_pred.eq(labels).sum().item()
        adv_imgs = attack_fn(model, imgs, labels, **attack_kwargs)
        with torch.no_grad():
            adv_out = model(adv_imgs)
            _, adv_pred = adv_out.max(1)
            correct_adv += adv_pred.eq(labels).sum().item()
        total += labels.size(0)
    orig_acc = 100. * correct_orig / total
    adv_acc  = 100. * correct_adv / total
    asr = 100. - adv_acc
    return asr, adv_acc, orig_acc

# ==================== 模块4：防御算法集 ====================
def adversarial_training(model, trainloader, testloader, device, epochs, epsilon, alpha, steps):
    """
    PGD对抗训练，批次内混合干净样本与对抗样本联合优化
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 35], gamma=0.1)
    best_adv_acc = 0.0
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f'AdvTrain Epoch {epoch+1}/{epochs}')
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            model.eval()
            adv_imgs = pgd_attack(model, imgs, labels, epsilon=epsilon, alpha=alpha, steps=steps)
            model.train()
            total_imgs = torch.cat([imgs, adv_imgs], dim=0)
            total_labels = torch.cat([labels, labels], dim=0)
            optimizer.zero_grad()
            outputs = model(total_imgs)
            loss = criterion(outputs, total_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        clean_acc = evaluate_clean(model, testloader, device)
        asr_pgd, _, _ = test_attack(model, testloader, pgd_attack, device,
                                    epsilon=epsilon, alpha=alpha, steps=steps)
        print(f'Epoch {epoch+1}: Clean Acc={clean_acc:.2f}%, Adv Acc={100-asr_pgd:.2f}%, ASR={asr_pgd:.2f}%')
        if (100 - asr_pgd) > best_adv_acc:
            best_adv_acc = 100 - asr_pgd
            torch.save(model.state_dict(), "models/resnet18_adv.pth")
    print(f'Best Adversarial Accuracy: {best_adv_acc:.2f}%')

class JPEGDefense:
    """
    JPEG压缩降噪防御，通过有损压缩过滤高频对抗扰动
    """
    def __init__(self, quality=75):
        self.quality = quality

    def __call__(self, images):
        imgs_np = images.cpu().detach().numpy().transpose(0,2,3,1)
        defended = []
        for img in imgs_np:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)
            compressed = Image.open(buffer)
            compressed_np = np.array(compressed).astype(np.float32) / 255.0
            defended.append(compressed_np)
        defended = np.stack(defended, axis=0).transpose(0,3,1,2)
        return torch.tensor(defended, device=images.device)

def test_jpeg_defense(model, loader, attack_fn, device, quality=75, **attack_kwargs):
    """JPEG防御后对抗样本攻击成功率评估"""
    model.eval()
    jpeg_defense = JPEGDefense(quality)
    correct_defended = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = attack_fn(model, imgs, labels,** attack_kwargs)
        purified = jpeg_defense(adv_imgs)
        with torch.no_grad():
            outputs = model(purified)
            _, pred = outputs.max(1)
            correct_defended += pred.eq(labels).sum().item()
        total += labels.size(0)
    def_adv_acc = 100. * correct_defended / total
    asr_def = 100. - def_adv_acc
    return asr_def, def_adv_acc

# ==================== 模块5：攻防评估与军备竞赛 ====================
def parameter_sweep_epsilon(model, loader, device, attack_fn, epsilons):
    """遍历不同扰动阈值epsilon，记录对应攻击成功率"""
    asrs = []
    for eps in epsilons:
        asr, _, _ = test_attack(model, loader, attack_fn, device, epsilon=eps)
        asrs.append(asr)
    return asrs

def parameter_sweep_pgd_steps(model, loader, device, epsilon, step_sizes, steps_list):
    """遍历PGD迭代步数，记录对应攻击成功率"""
    asrs = []
    for steps in steps_list:
        asr, _, _ = test_attack(model, loader, pgd_attack, device,
                                epsilon=epsilon, alpha=epsilon/4, steps=steps)
        asrs.append(asr)
    return asrs

# ==================== 模块6：可视化与脆弱性分析 ====================
def visualize_attacks(model, testloader, device, epsilon, alpha, steps, num_samples=5):
    """输出原图、三类对抗样本、放大扰动对比图"""
    model.eval()
    images, labels = next(iter(testloader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)
    fgsm_imgs = fgsm_attack(model, images, labels, epsilon)
    pgd_imgs = pgd_attack(model, images, labels, epsilon, alpha, steps)
    deepfool_imgs = deepfool_attack(model, images, max_iter=deepfool_max_iter, overshoot=deepfool_overshoot)
    with torch.no_grad():
        orig_preds = model(images).argmax(1)
        fgsm_preds = model(fgsm_imgs).argmax(1)
        pgd_preds = model(pgd_imgs).argmax(1)
        deep_preds = model(deepfool_imgs).argmax(1)
    fig, axes = plt.subplots(num_samples, 8, figsize=(16, 2.5*num_samples))
    if num_samples == 1:
        axes = axes[None, :]
    for i in range(num_samples):
        orig = images[i].cpu().permute(1,2,0).numpy()
        fgsm = fgsm_imgs[i].cpu().permute(1,2,0).numpy()
        pgd = pgd_imgs[i].cpu().permute(1,2,0).numpy()
        deep = deepfool_imgs[i].cpu().permute(1,2,0).numpy()
        fgsm_pert = (fgsm - orig) * 100
        pgd_pert  = (pgd - orig) * 100
        deep_pert = (deep - orig) * 100
        axes[i,0].imshow(np.clip(orig,0,1))
        axes[i,0].set_title(f'Original: {classes[labels[i]]}')
        axes[i,1].imshow(np.clip(fgsm,0,1))
        axes[i,1].set_title(f'FGSM: {classes[fgsm_preds[i]]}')
        axes[i,2].imshow(np.clip(pgd,0,1))
        axes[i,2].set_title(f'PGD: {classes[pgd_preds[i]]}')
        axes[i,3].imshow(np.clip(deep,0,1))
        axes[i,3].set_title(f'DeepFool: {classes[deep_preds[i]]}')
        axes[i,4].imshow(np.clip(fgsm_pert, -1, 1))
        axes[i,4].set_title('FGSM Pert x100')
        axes[i,5].imshow(np.clip(pgd_pert, -1, 1))
        axes[i,5].set_title('PGD Pert x100')
        axes[i,6].imshow(np.clip(deep_pert, -1, 1))
        axes[i,6].set_title('DeepFool Pert x100')
        axes[i,7].axis('off')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('plots/attack_visualization.png', dpi=200)
    plt.show()

def class_vulnerability_analysis(model, loader, device, attack_fn, epsilon, alpha, steps):
    """统计每个类别单独的攻击成功率，绘制柱状图"""
    model.eval()
    class_correct = torch.zeros(10, device=device)
    class_total = torch.zeros(10, device=device)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = attack_fn(model, imgs, labels, epsilon=epsilon, alpha=alpha, steps=steps)
        with torch.no_grad():
            adv_preds = model(adv_imgs).argmax(1)
        for c in range(10):
            mask = (labels == c)
            class_total[c] += mask.sum()
            class_correct[c] += adv_preds[mask].eq(labels[mask]).sum()
    asr_per_class = (100. * (1 - class_correct / class_total)).cpu().numpy()
    sorted_indices = np.argsort(asr_per_class)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_asr = [asr_per_class[i].item() for i in sorted_indices]
    plt.figure(figsize=(10,5))
    plt.bar(sorted_classes, sorted_asr, color='skyblue')
    plt.ylabel('Attack Success Rate (ASR %)')
    plt.title('Class Vulnerability (PGD Attack)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/class_vulnerability.png')
    plt.show()

def tsne_visualization(model, loader, device, attack_fn, epsilon, alpha, steps, num_samples=1000):
    """提取干净/对抗样本特征，t-SNE降维对比特征空间分布"""
    model.eval()
    features_clean = []
    features_adv = []
    labels_clean = []
    labels_adv = []
    sample_count = 0
    for imgs, labels in loader:
        if sample_count >= num_samples:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        batch = imgs.size(0)
        if sample_count + batch > num_samples:
            imgs = imgs[:num_samples - sample_count]
            labels = labels[:num_samples - sample_count]
        adv_imgs = attack_fn(model, imgs, labels, epsilon=epsilon, alpha=alpha, steps=steps)
        with torch.no_grad():
            _, feats_clean = model(imgs, return_features=True)
            _, feats_adv   = model(adv_imgs, return_features=True)
        features_clean.append(feats_clean.cpu().numpy())
        features_adv.append(feats_adv.cpu().numpy())
        labels_clean.append(labels.cpu().numpy())
        labels_adv.append(labels.cpu().numpy())
        sample_count += len(imgs)
    features_clean = np.vstack(features_clean)
    features_adv   = np.vstack(features_adv)
    labels_clean   = np.concatenate(labels_clean)
    labels_adv     = np.concatenate(labels_adv)
    all_features = np.vstack([features_clean, features_adv])
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    embeddings = tsne.fit_transform(all_features)
    n_clean = len(features_clean)
    clean_emb = embeddings[:n_clean]
    adv_emb = embeddings[n_clean:]
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    scatter = plt.scatter(clean_emb[:,0], clean_emb[:,1], c=labels_clean, cmap='tab10', alpha=0.5, s=10)
    plt.title('Clean Samples Feature Space')
    plt.subplot(1,2,2)
    plt.scatter(adv_emb[:,0], adv_emb[:,1], c=labels_adv, cmap='tab10', alpha=0.5, s=10)
    plt.title('Adversarial Samples Feature Space')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig('plots/tsne_decision_boundary.png', dpi=200)
    plt.show()

# ==================== 主执行流程 ====================
if __name__ == "__main__":
    print("===== 6A 白盒对抗攻防实验 =====")
    print("设备:", device)

    # 1. 基准干净模型训练或加载
    model_clean = ResNet18_CIFAR().to(device)
    if os.path.exists("models/resnet18_clean.pth"):
        model_clean.load_state_dict(torch.load("models/resnet18_clean.pth", map_location=device))
        print("加载已有干净模型")
        clean_acc = evaluate_clean(model_clean, testloader, device)
        print(f"干净模型准确率: {clean_acc:.2f}%")
    else:
        print("开始训练干净模型...")
        train_standard(model_clean, trainloader, num_epochs, device)
        model_clean.load_state_dict(torch.load("models/resnet18_clean.pth", map_location=device))

    # 2. 三类白盒攻击测试
    print("\n--- 1. FGSM攻击测试 ---")
    asr_fgsm, adv_acc_fgsm, _ = test_attack(model_clean, testloader, fgsm_attack, device, epsilon=epsilon)
    print(f"FGSM ε={epsilon:.4f} ASR={asr_fgsm:.2f}%, AdvAcc={adv_acc_fgsm:.2f}%")

    print("\n--- 2. PGD攻击测试 ---")
    asr_pgd, adv_acc_pgd, _ = test_attack(model_clean, testloader, pgd_attack, device,
                                          epsilon=epsilon, alpha=pgd_alpha, steps=pgd_steps)
    print(f"PGD ε={epsilon:.4f} steps={pgd_steps} ASR={asr_pgd:.2f}%, AdvAcc={adv_acc_pgd:.2f}%")

    print("\n--- 3. DeepFool攻击测试 ---")
    subset_indices = list(range(1000))
    subset_loader = DataLoader(Subset(testset, subset_indices), batch_size=32, shuffle=False)
    asr_deepfool, adv_acc_deepfool, _ = test_attack(model_clean, subset_loader, deepfool_attack, device,
                                                    max_iter=deepfool_max_iter, overshoot=deepfool_overshoot)
    print(f"DeepFool ASR (1000 samples)={asr_deepfool:.2f}%, AdvAcc={adv_acc_deepfool:.2f}%")

    # 3. 攻击参数敏感性分析
    print("\n--- Epsilon 扰动大小对ASR的影响 (FGSM & PGD) ---")
    eps_values = [0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1]
    asr_fgsm_eps = []
    asr_pgd_eps = []
    for eps in eps_values:
        asr1, _, _ = test_attack(model_clean, testloader, fgsm_attack, device, epsilon=eps)
        asr2, _, _ = test_attack(model_clean, testloader, pgd_attack, device, epsilon=eps, alpha=eps/4, steps=10)
        asr_fgsm_eps.append(asr1)
        asr_pgd_eps.append(asr2)
        print(f"ε={eps:.4f}: FGSM ASR={asr1:.2f}%, PGD ASR={asr2:.2f}%")
    plt.figure()
    plt.plot(eps_values, asr_fgsm_eps, marker='o', label='FGSM')
    plt.plot(eps_values, asr_pgd_eps, marker='s', label='PGD-10')
    plt.xlabel('Epsilon (perturbation size)')
    plt.ylabel('Attack Success Rate (%)')
    plt.title('Effect of Epsilon on ASR')
    plt.legend()
    plt.grid()
    plt.savefig('plots/epsilon_asr.png')
    plt.show()

    print("\n--- PGD 迭代次数影响 ---")
    steps_list = [1, 2, 5, 10, 20, 40]
    asr_pgd_steps = []
    for s in steps_list:
        asr, _, _ = test_attack(model_clean, testloader, pgd_attack, device, epsilon=epsilon, alpha=pgd_alpha, steps=s)
        asr_pgd_steps.append(asr)
        print(f"steps={s}: ASR={asr:.2f}%")
    plt.figure()
    plt.plot(steps_list, asr_pgd_steps, marker='o')
    plt.xlabel('PGD Iterations')
    plt.ylabel('Attack Success Rate (%)')
    plt.title('PGD Convergence: Iterations vs ASR')
    plt.grid()
    plt.savefig('plots/pgd_steps_asr.png')
    plt.show()

    # 4. 防御策略测试
    # 4.1 对抗训练
    if os.path.exists("models/resnet18_adv.pth"):
        model_adv = ResNet18_CIFAR().to(device)
        model_adv.load_state_dict(torch.load("models/resnet18_adv.pth", map_location=device))
        print("加载已有对抗训练模型")
    else:
        print("\n--- 开始对抗训练 ---")
        model_adv = ResNet18_CIFAR().to(device)
        adversarial_training(model_adv, trainloader, testloader, device,
                             epochs=adv_epochs, epsilon=epsilon, alpha=pgd_alpha, steps=7)
        model_adv.load_state_dict(torch.load("models/resnet18_adv.pth", map_location=device))
    clean_acc_adv = evaluate_clean(model_adv, testloader, device)
    asr_pgd_on_adv, adv_acc_pgd_on_adv, _ = test_attack(model_adv, testloader, pgd_attack, device,
                                                        epsilon=epsilon, alpha=pgd_alpha, steps=pgd_steps)
    print(f"对抗训练模型: 干净准确率={clean_acc_adv:.2f}%, PGD攻击下准确率={adv_acc_pgd_on_adv:.2f}%, ASR={asr_pgd_on_adv:.2f}%")

    # 4.2 JPEG压缩防御
    print("\n--- JPEG压缩防御评估 ---")
    asr_jpeg, def_acc_jpeg = test_jpeg_defense(model_clean, testloader, pgd_attack, device, quality=jpeg_quality,
                                               epsilon=epsilon, alpha=pgd_alpha, steps=pgd_steps)
    print(f"JPEG防御 (q={jpeg_quality}): PGD攻击ASR={asr_jpeg:.2f}%, 防御后准确率={def_acc_jpeg:.2f}%")
    correct_jpeg_clean = 0
    total = 0
    jpeg_def = JPEGDefense(quality=jpeg_quality)
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            purified = jpeg_def(imgs)
            outputs = model_clean(purified)
            _, pred = outputs.max(1)
            correct_jpeg_clean += pred.eq(labels).sum().item()
            total += labels.size(0)
    clean_acc_jpeg = 100. * correct_jpeg_clean / total
    print(f"JPEG压缩后干净样本准确率: {clean_acc_jpeg:.2f}% (下降 {evaluate_clean(model_clean, testloader, device) - clean_acc_jpeg:.2f}%)")

    # 5. 军备竞赛：多重启PGD自适应攻击
    print("\n--- 军备竞赛：攻击防御后模型 ---")
    def pgd_random_restart(model, images, labels, epsilon, alpha, steps, restarts=3):
        """多次随机起点PGD，选取损失最大对抗样本"""
        orig_loss = 0
        best_adv = images.clone()
        for _ in range(restarts):
            adv = pgd_attack(model, images, labels, epsilon, alpha, steps, random_start=True)
            with torch.no_grad():
                loss = F.cross_entropy(model(adv), labels)
            if loss > orig_loss:
                best_adv = adv
                orig_loss = loss.item()
        return best_adv
    asr_restart, adv_acc_restart, _ = test_attack(model_adv, testloader, pgd_random_restart, device,
                                                  epsilon=epsilon, alpha=pgd_alpha, steps=pgd_steps, restarts=3)
    print(f"多起点PGD (3次重启) 对抗训练模型 ASR={asr_restart:.2f}%, AdvAcc={adv_acc_restart:.2f}%")
    if asr_restart > asr_pgd_on_adv * 1.2:
        print("警告: 多起点PGD使ASR明显上升，可能存在梯度掩蔽问题。")
    else:
        print("防御模型表现出较好的鲁棒性，梯度掩蔽不明显。")

    # 6. 可视化分析模块执行
    print("\n--- 可视化对抗样本 ---")
    visualize_attacks(model_clean, testloader, device, epsilon, pgd_alpha, pgd_steps, num_samples=5)

    print("\n--- 类别脆弱性分析 ---")
    class_vulnerability_analysis(model_clean, testloader, device, pgd_attack, epsilon, pgd_alpha, pgd_steps)

    print("\n--- t-SNE决策边界可视化 ---")
    tsne_visualization(model_clean, testloader, device, pgd_attack, epsilon, pgd_alpha, pgd_steps, num_samples=500)

    print("\n===== 实验完成，图表已保存至 plots/ 目录 =====")