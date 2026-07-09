"""
6A组 白盒对抗样本攻防实验 - 评估脚本
包含：三类白盒攻击、两种防御、参数分析、军备竞赛验证、全量可视化
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from PIL import Image
import io
from data_utils import *
from model import ResNet18_CIFAR

# ==================== 白盒攻击算法集 ====================
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

# ==================== 防御算法集 ====================
class JPEGDefense:
    """JPEG压缩降噪防御，通过有损压缩过滤高频对抗扰动"""
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

        adv_imgs = attack_fn(model, imgs, labels, **attack_kwargs)
        purified = jpeg_defense(adv_imgs)

        with torch.no_grad():
            outputs = model(purified)
            _, pred = outputs.max(1)
            correct_defended += pred.eq(labels).sum().item()

        total += labels.size(0)

    def_adv_acc = 100. * correct_defended / total
    asr_def = 100. - def_adv_acc

    return asr_def, def_adv_acc

# ==================== 参数敏感性分析 ====================
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

# ==================== 提交要求的图表生成 ====================
def plot_confusion_matrix(model, loader, device, save_path='results/confusion_matrix.png'):
    """生成提交要求的分类混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Clean Samples)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.savefig('plots/confusion_matrix.png', dpi=200)
    plt.close()

def plot_comparison_table(results, save_path='results/comparison_table.png'):
    """生成提交要求的对比实验表格"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=results,
                     colLabels=['实验方法', '干净准确率(%)', '对抗准确率(%)', '攻击成功率(%)'],
                     loc='center', cellLoc='center',
                     colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    plt.title('不同攻击与防御方法性能对比', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.savefig('plots/comparison_table.png', dpi=200)
    plt.close()

# ==================== 可视化与脆弱性分析 ====================
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
    plt.savefig('results/attack_visualization.png', dpi=200)
    plt.close()

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
    plt.savefig('results/class_vulnerability.png')
    plt.close()

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
    plt.savefig('results/tsne_decision_boundary.png', dpi=200)
    plt.close()

# ==================== 主评估流程 ====================
if __name__ == "__main__":
    print("===== 6A 白盒对抗攻防实验 - 评估脚本 =====")
    print("设备:", device)

    trainloader, testloader = get_dataloaders()

    # 加载干净模型
    model_clean = ResNet18_CIFAR().to(device)
    model_clean.load_state_dict(torch.load("models/resnet18_clean.pth", map_location=device))
    print("✅ 加载干净模型成功")

    # 基础评估
    clean_acc = evaluate_clean(model_clean, testloader, device)
    print(f"干净模型准确率: {clean_acc:.2f}%")

    # 生成提交要求的混淆矩阵
    print("\n--- 生成混淆矩阵 ---")
    plot_confusion_matrix(model_clean, testloader, device)

    # 1. 三类白盒攻击测试
    print("\n--- 1. FGSM攻击测试 ---")
    asr_fgsm, adv_acc_fgsm, _ = test_attack(model_clean, testloader, fgsm_attack, device, epsilon=epsilon)
    print(f"FGSM ε={epsilon:.4f} ASR={asr_fgsm:.2f}%, AdvAcc={adv_acc_fgsm:.2f}%")

    print("\n--- 2. PGD攻击测试 ---")
    asr_pgd, adv_acc_pgd, _ = test_attack(model_clean, testloader, pgd_attack, device,
                                          epsilon=epsilon, alpha=pgd_alpha, steps=pgd_steps)
    print(f"PGD ε={epsilon:.4f} steps={pgd_steps} ASR={asr_pgd:.2f}%, AdvAcc={adv_acc_pgd:.2f}%")

    print("\n--- 3. DeepFool攻击测试 (1000样本) ---")
    subset_indices = list(range(1000))
    subset_loader = DataLoader(Subset(testset, subset_indices), batch_size=32, shuffle=False)
    asr_deepfool, adv_acc_deepfool, _ = test_attack(model_clean, subset_loader, deepfool_attack, device,
                                                    max_iter=deepfool_max_iter, overshoot=deepfool_overshoot)
    print(f"DeepFool ASR={asr_deepfool:.2f}%, AdvAcc={adv_acc_deepfool:.2f}%")

    # 2. 攻击参数敏感性分析
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
    plt.savefig('results/epsilon_asr.png')
    plt.close()

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
    plt.savefig('results/pgd_steps_asr.png')
    plt.close()

    # 3. 防御策略测试
    print("\n--- 对抗训练模型评估 ---")
    model_adv = ResNet18_CIFAR().to(device)
    model_adv.load_state_dict(torch.load("models/resnet18_adv.pth", map_location=device))

    clean_acc_adv = evaluate_clean(model_adv, testloader, device)
    asr_pgd_on_adv, adv_acc_pgd_on_adv, _ = test_attack(model_adv, testloader, pgd_attack, device,
                                                        epsilon=epsilon, alpha=pgd_alpha, steps=pgd_steps)
    print(f"对抗训练模型: 干净准确率={clean_acc_adv:.2f}%, PGD攻击下准确率={adv_acc_pgd_on_adv:.2f}%, ASR={asr_pgd_on_adv:.2f}%")

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
    print(f"JPEG压缩后干净样本准确率: {clean_acc_jpeg:.2f}% (下降 {clean_acc - clean_acc_jpeg:.2f}%)")

    # 4. 军备竞赛：多重启PGD自适应攻击
    print("\n--- 军备竞赛：多起点PGD攻击防御模型 ---")
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

    # 生成提交要求的对比表格
    print("\n--- 生成对比实验表格 ---")
    results = [
        ["干净基准模型", f"{clean_acc:.2f}", "-", "-"],
        ["FGSM攻击 (ε=16/255)", "-", f"{adv_acc_fgsm:.2f}", f"{asr_fgsm:.2f}"],
        ["PGD攻击 (ε=16/255, steps=20)", "-", f"{adv_acc_pgd:.2f}", f"{asr_pgd:.2f}"],
        ["DeepFool攻击", "-", f"{adv_acc_deepfool:.2f}", f"{asr_deepfool:.2f}"],
        ["对抗训练防御", f"{clean_acc_adv:.2f}", f"{adv_acc_pgd_on_adv:.2f}", f"{asr_pgd_on_adv:.2f}"],
        ["JPEG压缩防御 (q=75)", f"{clean_acc_jpeg:.2f}", f"{def_acc_jpeg:.2f}", f"{asr_jpeg:.2f}"]
    ]
    plot_comparison_table(results)

    # 5. 可视化分析模块执行
    print("\n--- 可视化对抗样本 ---")
    visualize_attacks(model_clean, testloader, device, epsilon, pgd_alpha, pgd_steps, num_samples=5)

    print("\n--- 类别脆弱性分析 ---")
    class_vulnerability_analysis(model_clean, testloader, device, pgd_attack, epsilon, pgd_alpha, pgd_steps)

    print("\n--- t-SNE决策边界可视化 ---")
    tsne_visualization(model_clean, testloader, device, pgd_attack, epsilon, pgd_alpha, pgd_steps, num_samples=500)

    print("\n===== 评估完成 =====")
    print("✅ 所有提交要求的图表已保存至 results/ 目录")
    print("✅ 包含：loss_curve.png、confusion_matrix.png、comparison_table.png")