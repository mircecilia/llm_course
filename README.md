# 大模型课程作业

本仓库收录八次实验的关键程序、未执行的 Notebook、依赖清单和运行说明。首次运行请从[总复现说明](复现说明.md)选定对应环境，再执行该实验的命令。实验五、六使用远端 Linux/GPU，其余实验使用 CPU。

| 实验 | 主题 | 入口 |
| --- | --- | --- |
| 01 | MLP 手写分类与控制变量对照 | [训练与学习率扫描](lab01_mlp/mlp_digits.py) |
| 02 | ConvLSTM 弹跳小球预测 | [单球 Notebook](lab02_convlstm/convlstm_bounce.ipynb)、[双球选做](lab02_convlstm/convlstm_extra_multiball.ipynb) |
| 03 | 字符级最小语言模型 | [训练与采样脚本](lab03_min_llm/min_llm.py)、[Notebook](lab03_min_llm/min_llm_experiments.ipynb) |
| 04 | Vector RAG、GraphRAG、WikiRAG | [固定基准](lab04_rag/hw4_rag.py)、[API 扩展](lab04_rag/hw4_rag_api.py)、[Notebook](lab04_rag/hw4_rag_experiments.ipynb) |
| 05 | SmolVLM QLoRA 与 DPO | [远端复现指南](lab05_smolvlm/README.md) |
| 06 | RSI 定位、自查与验证器 | [远端复现指南](lab06_rsi/README.md) |
| 07 | 白模转写实与头像融合 | [本地 CPU 复现指南](lab07_depth_cpu/README.md) |
| 08 | 字符级语言模型知识蒸馏 | [本地 CPU 复现指南](lab08_distill/README.md) |

实验一至四使用 Python 3.10 课程环境。实验五至八各用独立环境，版本、安装来源、运行顺序和输出位置见对应说明。实验四的 API 阶段会发出真实请求，密钥由使用者通过环境变量提供。实验八的真实语料选做读取实验三源码内置的古诗，克隆时保留两个实验目录。

## 提交前检查

从仓库根目录使用 Python 3.10 或更新版本运行：

~~~powershell
python .\check_repository.py
git diff --cached --check
git diff --cached --name-only
~~~

[检查器](check_repository.py)只读检查提交白名单、语法、Notebook 空输出、本地链接、依赖引用、Bash 换行和常见凭据形态。完整的运行验证按各实验指南执行。

Git 白名单仅开放源程序、复现资料和必要配置。历史输出、外部数据、模型权重、报告、课程资料、密钥及本地环境均保存在本地。新增源码须显式更新白名单，Notebook 在提交前清除输出和执行计数。
