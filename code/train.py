import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import ResNet18_CIFAR
from data_utils import trainloader, testloader, device, evaluate_clean, epsilon, pgd_alpha, pgd_steps, adv_epochs

# 引入攻击函数（用于对抗训练）
def pgd_attack(model, images, labels, epsilon, alpha, steps, random_start=True):
    adv_images = images.clone().detach()
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, 0, 1)
    for _ in range(steps):
        adv_images = adv_images.clone().detach().requires_grad_(True)
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            eta = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = torch.clamp(images + eta, 0, 1)
    return adv_images.detach()

def train_standard(model, trainloader, epochs, device):
    """标准干净模型训练"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
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

def adversarial_training(model, trainloader, testloader, device, epochs, epsilon, alpha, steps):
    """对抗训练防御，生成鲁棒模型resnet18_adv.pth"""
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
        # 临时评估PGD攻击效果
        asr_pgd, _, _ = test_attack_temp(model, testloader, device, epsilon, alpha, steps)
        print(f'Epoch {epoch+1}: Clean Acc={clean_acc:.2f}%, Adv Acc={100-asr_pgd:.2f}%, ASR={asr_pgd:.2f}%')
        if (100 - asr_pgd) > best_adv_acc:
            best_adv_acc = 100 - asr_pgd
            torch.save(model.state_dict(), "models/resnet18_adv.pth")
    print(f'Best Adversarial Accuracy: {best_adv_acc:.2f}%')

# 临时评估函数，仅对抗训练内部调用
def test_attack_temp(model, loader, device, epsilon, alpha, steps):
    model.eval()
    correct_adv = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = pgd_attack(model, imgs, labels, epsilon, alpha, steps)
        with torch.no_grad():
            adv_out = model(adv_imgs)
            _, adv_pred = adv_out.max(1)
            correct_adv += adv_pred.eq(labels).sum().item()
        total += labels.size(0)
    adv_acc  = 100. * correct_adv / total
    asr = 100. - adv_acc
    return asr, adv_acc, 0

if __name__ == "__main__":
    from data_utils import num_epochs
    model = ResNet18_CIFAR().to(device)
    print("开始标准干净模型训练")
    train_standard(model, trainloader, num_epochs, device)
    model.load_state_dict(torch.load("models/resnet18_clean.pth", map_location=device))
    print("开始对抗训练")
    adversarial_training(model, trainloader, testloader, device, adv_epochs, epsilon, pgd_alpha, 7)