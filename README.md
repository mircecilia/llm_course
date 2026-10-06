# 大模型课程作业

四个 CPU 实验的关键程序与[复现说明](复现说明.md)。Git 只保留源码、未执行的 Notebook 和运行指南，不包含实验结果、下载数据、模型权重、缓存、报告或 API 凭据。

## 目录

```text
lab01_mlp/
└── mlp_digits.py                  # MLP 对照、最优组合复测、学习率扫描
lab02_convlstm/
├── convlstm_bounce.ipynb           # 单球 ConvLSTM 与六组对照
└── convlstm_extra_multiball.ipynb  # 双球扩展选做
lab03_min_llm/
├── min_llm.py                     # 字符级 GPT、采样、重复惩罚
└── min_llm_experiments.ipynb       # 实验总览与结果读取
lab04_rag/
├── hw4_rag.py                     # 三种 RAG 固定基准、K 扫描与 RRF
├── hw4_rag_api.py                 # LLM 知识构建、生成与变体评测
├── hw4_rag_experiments.ipynb       # 两层实验总览
└── hw4_wiki_figure.py              # 固定 Wiki 条目的紧凑布局绘图
README.md
复现说明.md
.gitignore
```

## 运行入口

先选择已安装依赖的环境，再从仓库根目录手动运行：

```powershell
# 若本机已有课程使用的 myenv。
conda activate myenv

# 实验一（另用 --experiment lr_scan 运行扫描）。
python .\lab01_mlp\mlp_digits.py --experiment all

# 实验三。
python .\lab03_min_llm\min_llm.py --suite

# 实验四第一层，需事先准备本地嵌入模型。
python .\lab04_rag\hw4_rag.py --mode all
```

实验二使用对应 Notebook，在已准备好依赖的 Jupyter 内核中按顺序执行。实验四第二层需要自行配置 API 环境变量，会产生网络请求和费用；全部命令及运行顺序见[复现说明](复现说明.md)。不要盲目整本运行 Notebook 或覆盖已有结果。

## 本地文件与 Git

新结果保存到仓库根目录的 `lab01_outputs/`、`lab02_outputs/`、`lab02_extra_outputs/`、`lab03_outputs/`、`lab04_outputs/`，不纳入 Git。源码中的指南小语料常量和合成数据生成逻辑保留，以便复现；外部数据和运行产物不提交。

`.gitignore` 使用关键文件白名单。新增程序须显式补充白名单；不要用 `git add -f` 强制加入结果、报告或密钥。提交前清空 Notebook 输出，避免把图片、日志或真实回答嵌入代码仓库。
