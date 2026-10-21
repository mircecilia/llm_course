# 实验六 RSI 精准定位复现

必做：指南 15 题（8 loc、4 count、3 exist），基线、10 步 SFT、验证器纠错、两轮各 10 步 RSI。选做 A：同一组场景与初始化下改为 8×8 网格，独立训练并比较正确率与 IoU。共两组、60 个优化步，不训练检测器、不使用 LLM API，不写实验报告。

## 环境与执行

在远端新建 `llm_course_lab06`，程序放在其 `src/` 中，使用该目录专用 `.venv`。本次从实验五已验证的 site-packages **复制**依赖到新 venv，不共享可写包目录，不下载模型或环境、不改 base。基座读取原有 SmolVLM-256M-Instruct 快照，不复用实验五适配器。

若从零安装，须先按使用者的授权规则确认，再使用镜像，仅在新 venv 安装：

```bash
python3.10 -m venv /指定路径/llm_course_lab06/.venv
/指定路径/llm_course_lab06/.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r src/requirements.txt
```

完成依赖与本地模型准备后：

```bash
export LAB06_ROOT=/指定路径/llm_course_lab06
export LAB06_MODEL=/已有模型路径/SmolVLM-256M-Instruct
bash "$LAB06_ROOT/src/run_all.sh"
```

`run_all.sh` 先检查解析器、数据哈希，再做一次无优化器更新的模型前向/反向检查，然后 GPU0 跑 4×4、GPU1 跑 8×8。训练和评测均使用 FP32；LoRA r=8、alpha=16、dropout=0.05，AdamW lr=1e-4，batch=1。指南标题中的 QLoRA 不等于其参考代码进行了 4-bit 量化，本实验保留参考代码的 FP32 设置。

已有 `outputs/grid4` 或 `outputs/grid8` 时主程序拒绝覆盖。重跑请使用新实验根目录，不要删除旧结果。

### 授权追加训练

原始 10 步结果仍保存在 `outputs/grid4`、`outputs/grid8`。`run_extended.sh` 将种子 SFT 与每轮 RSI 步数分开配置，追加实验另存，不改变场景、种子、问题或判分规则：

```bash
# 参数依次为：种子 SFT 步数、每轮 RSI 步数、独立实验名。
bash "$LAB06_ROOT/src/run_extended.sh" 60 10 s60_r10
# 进一步追加也必须换实验名，保留前一次尝试。
bash "$LAB06_ROOT/src/run_extended.sh" 150 10 s150_r10
# 展示实际数据回流的版本：种子 60 步，两轮 RSI 各 30 步。
bash "$LAB06_ROOT/src/run_extended.sh" 60 30 s60_r30
```

每组从同一基座与 LoRA 初始化重新起训；4×4 与 8×8 的预算相同。输出在 `outputs/<实验名>/grid4`、`grid8`，日志在 `logs/<实验名>`。复核某次追加结果时给 `audit_results.py` 添加 `--run-name <实验名>`。默认 `run_all.sh` 仍复现指南的 10 步版本，不能把追加步数写成原始预算。

完整闭环过程优先查看 `s60_r30`；更长种子训练的对照查看 `s150_r10`。两者不能混填为同一次运行。比较全部预算：

```bash
"$LAB06_ROOT/.venv/bin/python" "$LAB06_ROOT/src/audit_results.py" --root "$LAB06_ROOT" --compare-runs baseline10 s60_r10 s150_r10 s60_r30
```

### 固定权重的反馈上下文对照

原反馈只说明候选格覆盖不足，却没有在工具重答时展示具体候选。`--tool-context candidate` 可回传模型自己的候选坐标及高亮图；不提供 GT 正确坐标，验证器仍在未标记的原图上运行。默认行为仍是原反馈，新增方式仅作为独立对照。

```bash
bash "$LAB06_ROOT/src/run_feedback_probes.sh"
```

此脚本分别复用 `s60_r30`、`s150_r10` 的 SFT 权重，先逐字核对重载后的 15 个原始回答，再比较反馈方式，优化器更新为 0。输出在 `outputs/feedback_<源实验名>/grid*/feedback_comparison.json`。该对照没有观察到额外答对，不能写成有效改进；训练数据回流的证据来自原反馈版 RSI 的真实自查纠错。

## 数据与评测口径

- 256×256 白底、4 色圆/方、轮廓宽度 4，与参考场景生成顺序一致。训练图 40 张 seed=2026，评测图 15 张 seed=42。
- 固定题集只使用前 8 张图定位、前 4 张图计数、前 3 张图存在性，种子样本是这 15 道题。这是指南规定的**同题回测**，不是独立测试泛化；40 张 train 图不擅自替换指南种子。
- 存在性题全部为正例；计数只看首个阿拉伯数字，存在性沿用指南字符串规则。保留原始答案，旁列复核规则局限，不改分母或事后修分。
- loc 正确率按指定 GT 框中心网格精确匹配；IoU 为预测网格与该 GT 框的面积交并比，解析失败记 0，平均分母固定为 8。
- 验证器按颜色连通块检查中心或 >=50% 的框面积覆盖，不识别形状，可能合并同色目标或接受另一个同色目标。主答案判分与验证器判断分别保存。
- 与指南一样，仅在 GT 判分为错时触发纠错，原本答对的题保留原答。因此这是 oracle-triggered 教学闭环，`由对变错` 按实现为 0，不当作无 GT 部署收益。

## 参考代码的必要补齐

1. 样板写 480px、`make_seed`，指南与代码写 256px、`make_sft`：采用 256px，并给命令提供别名。
2. RSI 从已经完成 10 步的 SFT 权重继续，不另起随机适配器。两轮 RSI 使用连续的 AdamW 优化器；种子阶段与 RSI 阶段的优化器分开。
3. 自查输出传给后续验证器，不丢弃 `pred2_self`。只有原答可解析时才生成高亮图；格式失败时由工具提示格式重答，不伪造模型预测网格。
4. 实际模型纠错答案同时通过解析式验证器和 GT 判分后才入池，不把 GT 标准答案直接写成模型产物。
5. 原实现每轮从池头取 10 条，会完全漏掉池末新增样本。现每轮先读新增样本，再按 seed=42+轮次打乱旧样本补足；记录新样本是否真正参加反传。种子 10 步仍保留原顺序（8 loc + 2 count）。
6. 保留每轮数据池、适配器、逐题输出、纠错分支和逐步 loss，不覆盖中间版本。

## 后续填表所需证据

| 要求 | 实际输出 |
|---|---|
| 环境、参数占比、真实日志 | `outputs/grid*/environment.json`、`parameters.json`、`logs/*.log` |
| 指南表 6 三阶段与 RSI 行 | `outputs/grid4/metrics.csv` |
| 指南表 7 两轮记录 | `outputs/grid4/round_metrics.csv`，初始规模 15 |
| 纠错触发、错转对、对转错、入池 | `sft_harness/metrics.json`、`rsi*_correction/metrics.json` |
| 原始题目、模型输出、工具证据 | 各阶段 `predictions.json`、`records.json`、高亮图 `overlays/` |
| 种子、模型产出、池增长 | `pool_round0/1/2.json`、`sft_data.json`、`rsi_memory.json` |
| 真实训练 loss 与新样本使用 | `*_train/history.csv`、`training.json` |
| 图表 | `rsi_curves.png`、`training_loss.png`、`outputs/grid_comparison.png` |
| 选做 A 对照 | `outputs/grid_comparison.csv` |
| 自动重算与公平性检查 | `outputs/audit.json` |
| 追加预算完整对照 | `outputs/extension_budget_comparison.csv/png` |
| 固定权重反馈对照 | `outputs/feedback_comparison.csv` |

追加版本表中的 `outputs/grid*` 均须替换为 `outputs/<实验名>/grid*`，不要误读原始 10 步的零定位结果。

代表性案例应从实际记录选取“直接答对、自查改对、验证器纠对、仍然失败”。某类没有发生时标明未观察到，不能手工生成模型回答。`自查改写`、`自查改对`、`工具调用`与`工具改对`是不同计数。

数据池规模按记录数统计。因为种子已含同题 GT，新增的模型纠错回放通常是同题重加权，不等于获得了同样多的新独立问题。计数的首数字判分可能接受“1、2、3”之类歧义回答，应另看原文，不能把指南正确率直接当成完整回答质量。

只复核现有结果，不训练：

```bash
"$LAB06_ROOT/.venv/bin/python" "$LAB06_ROOT/src/audit_results.py" --root "$LAB06_ROOT"
```

Git 只包含关键程序、Notebook、配置与复现说明；图像、结果、权重、运行日志和报告不提交，也不自动推送。
