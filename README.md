# 大模型课程作业一

本目录中的 `mlp_digits.py` 按实验指南实现了 digits 数据集上的 MLP 控制变量实验。
代码固定使用 CPU，不需要联网下载数据，也不会调用 `sklearn.neural_network.MLPClassifier`。

## 手动运行

本机已有的 `myenv` 环境包含实验指南要求的 `torch`、`scikit-learn`、`numpy` 和
`matplotlib`，本仓库不会自动安装依赖。由于当前终端中的 `python` 实际指向
`C:\python\python.exe`，请显式使用 `myenv` 的解释器。

在本目录的终端中手动运行全部实验：

```powershell
& 'C:\Users\Cecilia\.conda\envs\myenv\python.exe' .\mlp_digits.py --experiment all
```

也可以只运行单组实验，例如：

```powershell
& 'C:\Users\Cecilia\.conda\envs\myenv\python.exe' .\mlp_digits.py --experiment baseline
& 'C:\Users\Cecilia\.conda\envs\myenv\python.exe' .\mlp_digits.py --experiment exp1
& 'C:\Users\Cecilia\.conda\envs\myenv\python.exe' .\mlp_digits.py --experiment failure
& 'C:\Users\Cecilia\.conda\envs\myenv\python.exe' .\mlp_digits.py --experiment best
```

可选实验名称为 `baseline`、`exp1` 至 `exp7`、`failure` 和 `best`。运行后会生成：

- `result_*.png`：每组的训练损失和测试准确率曲线；
- `history_*.csv`：每个 epoch 的训练损失与测试准确率；
- `results.csv`：最终测试准确率、训练耗时和随机种子汇总。

`best` 使用固定的数据划分，并以预先指定的种子 42、43、44 各复测一次，汇报三次
最终准确率的均值与标准差。三次曲线分别保存为 `result_exp8_run1.png`、
`result_exp8_run2.png` 和 `result_exp8_run3.png`，与报告样板中的图 9 占位名称一致。
运行 `all` 时还会生成 `result_summary.png`，用于对比各组最终准确率和训练耗时。
所有表格数字和图片均应以你亲自在终端运行得到的结果为准。
