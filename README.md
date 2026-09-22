# 大模型课程作业

本仓库为大模型全部课程实验的代码仓库，仓库内容如下

## 仓库内容

- `mlp_digits.py`：全部模型、训练、评估和绘图代码；
- `convlstm_bounce.ipynb`：实验二 ConvLSTM 弹跳小球预测、对照实验与可视化；
- `convlstm_extra_multiball.ipynb`：实验二创新加分，双球时空预测扩展；
- `hw3_min_llm/min_llm.py`：实验三字符级最小 GPT、控制变量实验、采样与重复惩罚；
- `min_llm_experiments.ipynb`：实验三完整实验总览与报告数据索引；
- `复现说明.md`：环境要求、运行命令、输出文件和参考结果。

## 快速开始

环境依赖已准备好时，在仓库根目录运行：

```powershell
python .\mlp_digits.py --experiment all
```

完整复现流程和本实验使用的固定协议见[复现说明](复现说明.md)。

实验三完整运行：

```powershell
python .\hw3_min_llm\min_llm.py --suite
```

实验三会使用 CPU 和指南内置语料，产物单独保存到 `lab03_outputs/`，不会覆盖前两次实验。
