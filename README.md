# 基于卷积神经网络的图像分类实验项目

## 📜 项目概述
本项目是一个基于卷积神经网络（CNN）的图像分类实验，支持使用 **LeNet** 和 **ResNet18** 作为骨干网络进行图像分类任务。项目使用 **PyTorch** 和 **PyTorch Lightning** 构建，包含数据加载、模型训练、验证和测试等完整流程。

---

## 📂 目录结构
```
Users/ruanyudi/Documents/刘新龙实验二
├── .git/
├── .~卷积神经网络的图像分类实验报告模板.docx
├── Brodatz/
│   ├── test/
│   ├── train/
│   ├── val/
│   └── val.txt
├── README.md
├── __pycache__/
├── checkpoints/
│   └── best_model.ckpt
├── configs/
│   ├── lenet.yaml
│   └── resnet18.yaml
├── datasets/
│   ├── MyDatasets.py
│   └── __pycache__/
├── lightning_logs/
│   ├── version_0/
│   └── version_1/
├── models/
│   ├── LeNet/
│   ├── __pycache__/
│   ├── classifier.py
│   ├── lightning_base.py
│   └── resnet/
├── move_imgs.py
├── test.py
├── train.py
├── utils.py
└── 基于卷积神经网络的图像分类实验报告模板.docx
```

---

## 💻 环境依赖
本项目依赖以下环境与库：
- Python 3.x
- PyTorch
- PyTorch Lightning
- torchvision
- omegaconf
- matplotlib
- seaborn
- torchmetrics

### ✅ 安装依赖
```bash
pip install torch torchvision pytorch-lightning omegaconf matplotlib seaborn torchmetrics
```

---

## 🚀 使用方法

### 📃 配置文件
项目提供了两种配置文件，可在 `configs` 文件夹中找到：
- **configs/lenet.yaml**：使用 LeNet 模型
- **configs/resnet18.yaml**：使用 ResNet18 模型

可根据需要修改配置文件中的参数，如 **训练轮数、学习率、批量大小** 等。

### 🟢 训练模型
```bash
python train.py --config configs/resnet18.yaml
```
通过 `--config` 参数指定配置文件。

### 🔵 测试模型
```bash
python test.py --config configs/resnet18.yaml --ckpt_path ./checkpoints/best_model.ckpt
```
- `--config`：指定配置文件
- `--ckpt_path`：指定模型 checkpoint 文件

### 🟡 绘制混淆矩阵
测试阶段自动绘制混淆矩阵并保存为 `confusion_matrix.png`。


---

## 🤝 贡献
欢迎提交 **issue** 或 **pull request**，感谢您的支持与贡献！

---

🎉 感谢使用本项目，希望对您的学习和研究有所帮助！