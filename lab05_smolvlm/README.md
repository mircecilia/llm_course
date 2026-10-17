# 实验五 SmolVLM QLoRA 与 DPO 复现

本实验只用指南的合成几何图像，不需要个人数据或 LLM API。模型为 `HuggingFaceTB/SmolVLM-256M-Instruct`。先读 `protocol.json`，再执行命令；不根据评测表现改种子、问题、图像或判分规则。

## 隔离目录与环境

远端实验目录由 `LAB05_ROOT` 指定，以下以当前账户主目录下 `llm_course_lab05` 为例，独立环境为该目录下 `.venv`。Python 3.10 来自既有解释器创建的标准 venv，未使用 `--system-site-packages`，不修改原解释器、base、其他环境或系统 CUDA。模型、pip/HF 缓存、日志和输出均位于该目录。

安装时只对当前命令设置镜像，不修改全局 pip/conda 配置：

```bash
export LAB05_ROOT="$HOME/llm_course_lab05"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_CACHE_DIR="$LAB05_ROOT/cache/pip"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$LAB05_ROOT/cache/huggingface"
export HF_HUB_DISABLE_XET=1
```

关键依赖固定为 torch 2.6.0、torchvision 0.21.0、transformers 4.57.6、peft 0.17.1、accelerate 1.10.1、bitsandbytes 0.48.2、huggingface_hub 0.36.0、NumPy 2.2.6、Pillow 11.3.0、Matplotlib 3.10.7。完整实际版本另存 `outputs/requirements.lock.txt`。模型 commit 在 `models/revision.txt`，训练及评测均离线读取同一模型快照。

`requirements.txt` 只固定上述直接依赖，不冒充完整锁文件。在新的远端实验目录中，先把本目录关键程序、配置与依赖清单复制到 `src/`。仅首次准备环境时执行以下命令（已有 `.venv` 则跳过，不重建）：

```bash
# 使用已有的 Python 3.10 创建新 venv，不安装到该解释器原来的环境。
python3.10 -m venv "$LAB05_ROOT/.venv"
"$LAB05_ROOT/.venv/bin/python" -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r "$LAB05_ROOT/src/requirements.txt"
```

已验证的模型 revision 为 `7e3e67edbbed1bf9888184d9df282b700a323964`。下载模型时固定此版本，并将完整快照放入 `$LAB05_ROOT/models/SmolVLM-256M-Instruct/`；不把权重提交 Git。

## 必做与选做

- 基线：P01–P15，count/attr/exist 各 5 题，逐题保存原始回答。
- SFT：NF4 4-bit 基座、float32 计算，LoRA r=8/alpha=16/dropout=0.05，batch=1、60 个真实优化步、lr=1e-4。
- DPO：关闭适配器的冻结 NF4 基座为 reference，答案 token 总 log-probability，beta=0.1、40 步、lr=5e-5。先用原 12 对，再加入 5 个自拟对重新从同一 SFT 开始。
- DPO 开启非重入梯度检查点以降低两条 policy 计算图的激活显存，训练模式下显式将 Dropout 置零，reference 在 eval/no_grad 下预计算；不以减少步数、改图像或改数据规避显存限制。
- 任务五(a)：17 对偏好数据下 beta=0.05/0.1/0.5 各 40 步，0.1 复用 dpo17。不同 beta 的初始化适配器、seed=42 和样本顺序完全一致。不做另一个二选一的 r 扫描。
- 三阶段评测统一采用 fp32 基座和独立加载的单个适配器，防止精度不同或重复叠加适配器。解码均为 greedy、最多 24 个新 token。

指南 SFT 的 15 条数据就是这 15 道评测题。这里是同题回测，不宣称独立测试集泛化；新增 5 题也参与增强 DPO，只用于观察相关失败模式是否受抑制。

## 手动分步执行

先将本目录源码放到远端新目录的 `src/`，准备好镜像下载的模型。以下命令不会调用 API：

```bash
cd "$HOME/llm_course_lab05"
export LAB05_ROOT="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$LAB05_ROOT/cache/huggingface"
export MPLCONFIGDIR="$LAB05_ROOT/cache/matplotlib"
export CUDA_VISIBLE_DEVICES=0
PY="$LAB05_ROOT/.venv/bin/python"

"$PY" src/hw5_dpo.py --mode make_data
"$PY" src/hw5_dpo.py --mode check
"$PY" src/hw5_dpo.py --mode infer
"$PY" src/hw5_dpo.py --mode sft --run sft --steps 60
"$PY" src/hw5_dpo.py --mode eval --run sft
"$PY" src/hw5_dpo.py --mode dpo --run dpo12 --pairs 12 --steps 40 --beta 0.1
"$PY" src/hw5_dpo.py --mode eval --run dpo12
"$PY" src/hw5_dpo.py --mode dpo --run dpo17 --pairs 17 --steps 40 --beta 0.1
"$PY" src/hw5_dpo.py --mode eval --run dpo17
"$PY" src/hw5_dpo.py --mode dpo --run beta005 --pairs 17 --steps 40 --beta 0.05
"$PY" src/hw5_dpo.py --mode eval --run beta005
"$PY" src/hw5_dpo.py --mode dpo --run beta05 --pairs 17 --steps 40 --beta 0.5
"$PY" src/hw5_dpo.py --mode eval --run beta05
"$PY" src/hw5_dpo.py --mode summarize

"$PY" src/hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1
"$PY" src/hw5_dpo.py --dpo_check 1.5,1.0,0.2,0.6,0.1
"$PY" src/hw5_dpo.py --dpo_check 0.8,1.2,0.9,0.6,0.1
```

每个运行目录只允许首次创建，已有结果时直接报错，不覆盖。复跑请使用新的 `LAB05_ROOT`，复制模型目录或使用模型绝对路径；不要删除原始实验产物。

若要两卡运行，首次实验可执行 `bash src/run_all.sh`；该脚本会先完成基线和 SFT，再在两卡上并行独立的 DPO 及 beta 对照。远端后台运行可用 `nohup bash src/run_all.sh > logs/runner.log 2>&1 &`，断开 SSH 不会使训练中断，结果仍写入新目录。状态见 `logs/status.txt`。

本次初次 DPO 在首个优化步前因显存不足退出。原始 `dpo12/`、`dpo17/` 及日志保留，启用梯度检查点后使用 `bash src/run_all.sh --resume-dpo` 从已完成的 SFT 继续，成功运行目录为 `dpo12_gc/`、`dpo17_gc/`。beta 扫描使用相同节省显存的实现；状态见 `logs/resume_status.txt`。不要在已有结果目录再次执行。

FP32 主评测结束后，为检查 NF4 训练到 FP32 评测切换的影响，先对 baseline、sft、dpo17_gc 使用相同问题、gold、判分和解码参数做 NF4 推理诊断，再扩展到全部已训练模型，避免只展示有利的阶段：`--mode eval --run <阶段> --eval-quantization nf4`。结果单独保存 `outputs/evaluations_nf4/`，不替换冻结的 FP32 主结果，也不重新训练。不能只凭 loss/正确率背离就把退化全部归因于 DPO 过度优化。

## 记录与可视化

- `data_manifest.json`：全部问题、gold、图像规格、边界框、原始 12 对与新增 5 对。
- `image_contact_sheet.png`：15 张真实输入图像，保持指南随机位置与可能的重叠。
- `environment.json` / `implementation_checks.json`：GPU、依赖版本、NF4、mask、参考模型、梯度符号检查。
- `sft/`、`dpo12/`、`dpo17/`、`beta005/`、`beta05/`：配置、完整逐步 loss/梯度日志，DPO 另存 margin 和四项序列 log-probability。
- `evaluations/<stage>/`：逐题原始回答、主判分、指南子串旁列、分类正确率与耗时。
- `runtime.json` 的 `total_s` 包含模型加载及 20 题（15 道内置 + 5 道自拟）评测、结果写入，不是仅 15 题纯推理时间。逐题 `inference_s` 才是各次回答的计时，两种口径不能混用。
- `all_metrics.csv`、`accuracy_comparison.png`、`training_curves.png`：真实记录汇总，绘图不另造数据。
- `dpo_manual_checks.csv`：三组手算复核。margin=0.01 时 loss=0.68815968，指南写的约 0.693 不够精确；原参考手算函数还存在负号错误。
- `faithfulness_9_review.csv`：baseline/SFT/dpo17 各取 P01、P06、P12。人工评分留空，AI 建议分与人工最终分分列，不把正确率冒充忠实度。
- `faithfulness_9_codex_review.csv`、`faithfulness_9_nf4_codex_review.csv`：两套精度各九条 CODEX 建议评分，本人评分保持空白。无意义片段的评分置信度低，不能冒充忠实拒答。
- `failure_candidates.csv`：所有真实判错回答。没有足够某类型失败时如实记录，不能编造案例。
- `experiment_audit.json`、`training_summary.csv`：运行 `python src/audit_results.py --outputs outputs`，复算逐题判分、全部 220 个优化步、DPO 损失公式，并核对 beta 扫描的样本顺序、起始似然、适配器和预算。
- `precision_comparison.csv/png`、`all_metrics_nf4.csv`、`accuracy_nf4.png`：与 FP32 主结果分列的 NF4 精度诊断。
- `failure_case_evidence.csv`：依据原始回答和图像规格列出的候选案例，包含颜色绑定有歧义的情形，不把候选行一律宣称为失败。
- `scoring_review_notes.csv`：短否定“没。”在两个 E03 回答中被固定规则漏判，另列 CODEX 语义复核意见，不回写冻结指标。因此自动表中自拟题的单题差异也可能是判分覆盖不足，不宜直接当作行为提升。

主判分冻结为中英文宽松语义匹配，先识别存在性否定；多个互相冲突的数字/形状不算正确。原指南的子串规则单独保存，以便解释 `有` 匹配 `没有` 等问题，不混用两套正确率。

## 无 GPU 时的复核与归档

复核已保存结果不需要模型、GPU 或 API。直接对照 `outputs/images/`、各阶段 `predictions.json`、原始训练日志和固定配置即可；不要为填写报告重新训练。九条忠实度评分应同时记录评分依据和评审身份，CODEX 复核意见不能自动填成“本人已人工确认”。无意义片段应单独标注为不可用回答，不能视为高忠实拒答。

`audit_results.py` 不训练、不改原始预测，但会重写汇总 CSV/JSON 和派生图片。需要保留全部旧产物时，先复制结果到新的复核目录，再对副本运行；仅阅读报告时无需执行它。

Git 仅保留本目录的关键 Python、未执行 Notebook、固定配置、运行脚本、直接依赖清单与本说明。不提交输出、模型、环境、报告、用户名或 SSH 地址。远端实验目录及失败日志原位保留，不执行清空或覆盖。

## 实现依据

模型加载方式参考 [SmolVLM 官方模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)。量化基座和 LoRA 训练参考 [PEFT 量化文档](https://huggingface.co/docs/peft/developer_guides/quantization)。DPO 损失及参考策略关系参考 [DPO 原论文](https://arxiv.org/abs/2305.18290)。课程要求的参考模型为禁用适配器的基座，按指南实现，不换成冻结 SFT 模型。
