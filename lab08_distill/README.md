# 实验八 自制语言模型知识蒸馏

按照实验指南第七节实现字符级 Transformer、固定种子合成语料与 CPU 训练。主对照固定教师 200 步、学生 120 步、batch 24、学习率 0.003、T=4、α=0.5。两组学生的初始化哈希与批次采样种子会保存供复核。消融用同一批 32 个缓存批次循环训练，每格 80 步。教师规模扫描使用 96、160、256 三个宽度和相同的教师训练步数。

本次使用 Windows、Python 3.12.7、PyTorch 2.5.1+cpu 和 Matplotlib 3.10.9。保留完整仓库目录，拓展 F 需要读取同级实验三源码。用 PowerShell 7 从本目录建立独立环境并手动执行：

```powershell
python -m venv .venv
$lab08Python = '.\.venv\Scripts\python.exe'
$env:PIP_CACHE_DIR = Join-Path (Get-Location) 'cache\pip'
& $lab08Python -m pip install -r .\requirements.txt `
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple `
    --extra-index-url https://download.pytorch.org/whl/cpu
& $lab08Python -m pip check
.\run_all.ps1
```

CPU wheel 安装方式来自 [PyTorch 历史版本说明](https://docs.pytorch.org/get-started/previous-versions/#v2-5-1)。如使用其他版本，记录实际版本并将结果另存。

完整入口依次执行 `quick`、`teacher --retrain`、`all`、`ablate`、`scale`，再运行拓展脚本。入口要求输出目录尚不存在。已有默认结果时，给新运行指定独立目录：

```powershell
.\run_all.ps1 -Output .\runs_distill_repeat
```

只想先自检时，给自检单独指定目录，以便随后仍能使用完整入口的默认目录：

```powershell
& $lab08Python .\distill_lm.py --mode quick --output .\runs_distill_smoke
```

也可以单独运行各阶段。`teacher`、`all`、`ablate`、`scale` 和 `run_extensions.py` 须使用同一个 `--output`，且先生成教师检查点和主对照。

`runs_distill/` 下的 `results_teacher.json`、`results_all.json`、`results_ablate.json`、`results_scale.json` 保存实测与训练记录。阶段总耗时见 `phases.json`，单模型优化耗时见每行的 `time_s`。`curves.png` 的 CE 和混合目标量纲不同，模型比较使用共同的留出集 CE。表 8 的三段样例均使用提示“月”，采样种子 31415、temperature 0.9、top-k 20，报告仅显示新生成的 60 字。

复现保留了指南的语料生成方式。训练和留出合成句种子分别为 0 与 999，但两边共享 16 首固定古诗，且合成模板可能产生重复句。留出损失衡量该协议下的比较结果，不能据此证明对完全不重叠文本的泛化。教师和学生的词嵌入与输出头不共享权重。依赖实际快照、运行结果、权重和报告留在本地。

## 拓展 A–F

`ablate` 生成 B 的 4×4 热力图，`scale` 生成 E 的三档教师规模对照。完成主对照后，也可以单独复跑其余四项：

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_extensions.py'
```

- **A，FitNet**：在原蒸馏目标上加入权重 0.1 的隐状态 MSE，以 64→160 线性投影匹配最后隐状态。复用相同初值、批次顺序和 120 步下的 `feat=0` 结果。投影只在训练时使用。
- **C，自蒸馏**：把相同 64 维模型用硬标签训练 360 步作为教师，以新初始化的 120 步学生与从零学生比较。
- **D，动态量化**：将主对照蒸馏模型的线性层动态量化为 INT8，比较同格式权重文件大小和重载后的留出 CE。CPU 延迟用同一输入 `1×96` 和主对照的线程数（本次为 8），两种模型各预热 20 次、正式各测 60 次，交替执行并取中位数。
- **F，真实语料**：从[实验三源码](../lab03_min_llm/min_llm.py)读取 72 条内置古诗文本，按整条及相关片段以种子 999 划分为 55 条训练、17 条留出。词表只由训练文本和未知字符标记组成。两边各自比较从零到蒸馏的留出 CE，真实语料与合成语料的绝对 CE 受词表、文本和划分差异影响。

新增数据、导出权重及 KL 归一化核验见 `runs_distill/extensions/`，日志见 `runs_distill/extensions.log`。单独调用会跳过已有结果，各阶段完成后独立写入 JSON。
