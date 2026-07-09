6A组 白盒对抗样本攻防实验
1. 环境配置
Python >=3.8，GPU支持CUDA加速，依赖包见 requirements.txt
2. code文件夹文件功能说明
model.py：ResNet18_CIFAR网络、归一化层模型定义
data_utils.py：CIFAR10数据集加载、数据增强、全局常量、工具定义
train.py：标准干净模型训练、对抗训练脚本，输出resnet18_clean.pth / resnet18_adv.pth权重
eval.py：FGSM/PGD/DeepFool攻击、JPEG防御、指标评估、参数扫描、可视化绘图、t-SNE分析、军备竞赛自适应攻击
demo.ipynb：交互式分步演示，加载模型、生成对抗样本、可视化对比
3. 完整运行步骤
安装依赖：打开终端执行
pip install -r requirements.txt
分步运行
① 训练干净模型：python train.py
② 对抗训练鲁棒模型：python train.py
③ 完整攻防评估+绘图：python eval.py
输出文件
模型权重自动保存至上级models文件夹
所有折线图、样本可视化、t-SNE图自动保存至上级plots文件夹
4. 项目根目录权重说明
resnet18_clean.pth：原始干净ResNet18模型权重
resnet18_adv.pth：对抗训练防御鲁棒模型权重
