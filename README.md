# 大模型课程作业

本仓库为大模型全部课程实验的代码仓库，仓库内容如下

## 仓库内容

- `mlp_digits.py`：全部模型、训练、评估和绘图代码；
- `convlstm_bounce.ipynb`：实验二 ConvLSTM 弹跳小球预测、对照实验与可视化；
- `convlstm_extra_multiball.ipynb`：实验二创新加分，双球时空预测扩展；
- `min_llm.py`：实验三字符级最小 GPT、控制变量实验、采样与重复惩罚；
- `min_llm_experiments.ipynb`：实验三完整实验总览与报告数据索引；
- `hw4_rag.py`：实验四三种 RAG 检索、20 题评测、K 扫描与 RRF 选做；
- `hw4_rag_api.py`：实验四真实 API 抽取、提示词迭代、九个回答与独立补充评测；
- `hw4_rag_experiments.ipynb`：实验四可运行总览及报告数据索引；
- `复现说明.md`：环境要求、运行命令、输出文件和参考结果。

## 快速开始

环境依赖已准备好时，在仓库根目录运行：

```powershell
python .\mlp_digits.py --experiment all
```

完整复现流程和本实验使用的固定协议见[复现说明](复现说明.md)。

实验三完整运行：

```powershell
python .\min_llm.py --suite
```

实验三会使用 CPU 和指南内置语料，产物单独保存到 `lab03_outputs/`，不会覆盖前两次实验。

实验四在已准备好本地嵌入模型的 `myenv` 中运行：

```powershell
& 'C:\Users\Cecilia\.conda\envs\myenv\python.exe' .\hw4_rag.py --mode all
```

实验四主检索产物写入 `lab04_outputs/`；API 补充另存 `lab04_outputs/api_20261005/`，不替换固定 40 条图谱、9 个条目和原检索指标。密钥仅通过环境变量读取，不提交报告或凭据。运行命令见[复现说明](复现说明.md)。
