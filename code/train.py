"""
6A组 白盒对抗样本攻防实验 - 训练脚本
包含：标准干净模型训练、PGD对抗训练、训练曲线生成
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_utils import *
from model import ResNet18_CIFAR
from eval import pgd_attack, test_attack

def train_standard(model, trainloader, testloader, epochs, device):
    """标准监督训练，保存测试集最优权重"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate,
        momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 80, 110], gamma=0.1
    )

    best_acc = 0.0
    model.train()

    # 保存训练曲线数据
    train_losses = []
    test_accs = []

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

        # 评估并保存最优模型
        acc = evaluate_clean(model, testloader, device)
        print(f'Epoch {epoch+1}: Test Accuracy = {acc:.2f}%')

        # 记录训练数据
        train_losses.append(running_loss/len(trainloader))
        test_accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/resnet18_clean.pth")

    # 生成训练曲线
    plot_training_curve(train_losses, test_accs)
    print(f'Best Clean Accuracy: {best_acc:.2f}%')
    return best_acc

def adversarial_training(model, trainloader, testloader, device, epochs, epsilon, alpha, steps):
    """PGD对抗训练，批次内混合干净样本与对抗样本联合优化"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.1,
        momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 25, 35], gamma=0.1
    )

    best_adv_acc = 0.0
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f'AdvTrain Epoch {epoch+1}/{epochs}')

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # 生成对抗样本
            model.eval()
            adv_imgs = pgd_attack(model, imgs, labels, epsilon=epsilon, alpha=alpha, steps=steps)
            model.train()

            # 混合数据联合训练
            total_imgs = torch.cat([imgs, adv_imgs], dim=0)
            total_labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad()
            outputs = model(total_imgs)
            loss = criterion(outputs, total_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss/(pbar.n+1))

        scheduler.step()

        # 评估
        clean_acc = evaluate_clean(model, testloader, device)
        asr_pgd, _, _ = test_attack(model, testloader, pgd_attack, device,
                                    epsilon=epsilon, alpha=alpha, steps=steps)

        print(f'Epoch {epoch+1}: Clean Acc={clean_acc:.2f}%, Adv Acc={100-asr_pgd:.2f}%, ASR={asr_pgd:.2f}%')

        if (100 - asr_pgd) > best_adv_acc:
            best_adv_acc = 100 - asr_pgd
            torch.save(model.state_dict(), "models/resnet18_adv.pth")

    print(f'Best Adversarial Accuracy: {best_adv_acc:.2f}%')
    return best_adv_acc

def plot_training_curve(train_losses, test_accs):
    """生成提交要求的训练损失与准确率曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Curve')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/loss_curve.png', dpi=200)
    plt.savefig('plots/loss_curve.png', dpi=200)
    plt.close()

if __name__ == "__main__":
    print("===== 6A 白盒对抗攻防实验 - 训练脚本 =====")
    print("设备:", device)

    trainloader, testloader = get_dataloaders()

    # 1. 基准干净模型训练或加载
    model_clean = ResNet18_CIFAR().to(device)
    if os.path.exists("models/resnet18_clean.pth"):
        model_clean.load_state_dict(torch.load("models/resnet18_clean.pth", map_location=device))
        print("加载已有干净模型")
        clean_acc = evaluate_clean(model_clean, testloader, device)
        print(f"干净模型准确率: {clean_acc:.2f}%")
    else:
        print("开始训练干净模型...")
        train_standard(model_clean, trainloader, testloader, num_epochs, device)

    # 2. 对抗训练
    if os.path.exists("models/resnet18_adv.pth"):
        print("加载已有对抗训练模型")
    else:
        print("\n--- 开始对抗训练 ---")
        model_adv = ResNet18_CIFAR().to(device)
        adversarial_training(model_adv, trainloader, testloader, device,
                             epochs=adv_epochs, epsilon=epsilon, alpha=pgd_alpha, steps=7)

    print("\n训练完成，模型已保存至 models/ 目录")
    print("训练曲线已保存至 results/ 目录")