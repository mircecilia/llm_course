# 大模型课程作业一

本仓库使用 PyTorch 在 `sklearn.datasets.load_digits` 数据集上实现多层感知器分类，
通过控制变量比较激活函数、网络深度、优化器、学习率、L2 正则化、Dropout 和
BatchNorm，并包含学习率发散实验与最优组合三次复测。

实验仅使用 CPU，数据集由 scikit-learn 内置提供，不需要联网下载数据，也没有调用
`sklearn.neural_network.MLPClassifier` 代替 PyTorch 训练流程。

## 仓库内容

- `mlp_digits.py`：全部模型、训练、评估和绘图代码；
- `复现说明.md`：环境要求、运行命令、输出文件和参考结果；
- `2024311031_徐梓源_实验作业一.docx`：实验报告。

## 快速开始

环境依赖已准备好时，在仓库根目录运行：

```powershell
python .\mlp_digits.py --experiment all
```

完整复现流程和本实验使用的固定协议见[复现说明](复现说明.md)。
