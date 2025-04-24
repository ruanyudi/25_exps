# ResNet50 迁移学习实验

## 实验背景
本实验基于PyTorch框架，使用ResNet50预训练模型进行迁移学习，实现对图像数据的分类任务。通过冻结Backbone层并自定义分类器，探索在小样本数据集上的模型适应能力。

## 环境要求
```bash
# 创建conda环境
conda create -n resnet50 python=3.9
conda activate resnet50

# 安装核心依赖
pip install torch torchvision pytorch-lightning omegaconf
```

## 数据集准备
1. 下载UCM数据集
2. 按以下结构组织文件：
```
UCM/
├── train/
│   ├── class1/
│   └── class2/
├── val/
└── test/
```

## 配置文件说明（configs/resnet50.yaml）
```yaml
Backbone:
  module_name: models.resnet.ResNet
  class_name: resnet50
  use_pretrained: True   # 启用预训练权重
  freeze: True           # 冻结Backbone层

Datasets:
  train_path: UCM/train  # 训练集路径
  Batch_size: 1          # 批大小

Parameters:
  max_epochs: 100        # 最大训练轮次
  learning_rate: 0.001   # 初始学习率

Classifier:
  num_classes: 10        # 输出类别数
  in_features_dim: 2048  # Backbone输出维度
```

## 训练执行
```bash
python train.py --config configs/resnet50.yaml
```

## 参考文献
1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)