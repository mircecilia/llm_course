"""作业一：在 sklearn digits 数据集上进行 MLP 控制变量实验。"""

import argparse
import csv
import time
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA_SPLIT_SEED = 42
BEST_MODEL_SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class Experiment:
    group: str
    name: str
    change: str
    filename: str
    hidden: tuple[int, ...] = (32,)
    activation: str = "sigmoid"
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    weight_decay: float = 0.0
    dropout: float = 0.0
    use_batch_norm: bool = False
    epochs: int = 30


BASELINE = Experiment(
    group="0",
    name="基线 Baseline",
    change="1x32 Sigmoid + SGD(0.1)",
    filename="baseline",
)

EXPERIMENTS = {
    "baseline": BASELINE,
    "exp1": replace(
        BASELINE,
        group="1",
        name="换激活函数",
        change="Sigmoid -> ReLU",
        filename="exp1",
        activation="relu",
    ),
    "exp2": replace(
        BASELINE,
        group="2",
        name="加深网络",
        change="1x32 -> 2x128",
        filename="exp2",
        hidden=(128, 128),
    ),
    "exp3": replace(
        BASELINE,
        group="3",
        name="换优化器",
        change="SGD(0.1) -> Adam(0.01)",
        filename="exp3",
        optimizer="adam",
        learning_rate=0.01,
    ),
    "exp4": replace(
        BASELINE,
        group="4",
        name="调学习率",
        change="0.1 -> 0.01",
        filename="exp4",
        learning_rate=0.01,
    ),
    "exp5": replace(
        BASELINE,
        group="5",
        name="加 L2 正则",
        change="weight_decay: 0 -> 1e-4",
        filename="exp5",
        weight_decay=1e-4,
    ),
    "exp6": replace(
        BASELINE,
        group="6",
        name="加 Dropout",
        change="dropout: 0 -> 0.2",
        filename="exp6",
        dropout=0.2,
    ),
    "exp7": replace(
        BASELINE,
        group="7",
        name="加 BatchNorm",
        change="每个隐藏层后加入 BatchNorm1d",
        filename="exp7",
        use_batch_norm=True,
    ),
    "best": Experiment(
        group="8",
        name="最优组合",
        change="ReLU + 2x128 + BatchNorm + Adam(0.001)",
        filename="exp8",
        hidden=(128, 128),
        activation="relu",
        optimizer="adam",
        learning_rate=0.001,
        use_batch_norm=True,
    ),
    "failure": replace(
        BASELINE,
        group="失败",
        name="学习率过大",
        change="learning_rate: 0.1 -> 10.0",
        filename="lr_too_big",
        learning_rate=10.0,
    ),
}


class MLP(nn.Module):
    def __init__(
        self,
        hidden: tuple[int, ...],
        activation: str,
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        activation_layer = {
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }[activation]

        layers: list[nn.Module] = []
        previous_width = 64
        for width in hidden:
            layers.append(nn.Linear(previous_width, width))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(activation_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_width = width
        layers.append(nn.Linear(previous_width, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels = load_digits(return_X_y=True)
    features = features.astype(np.float32) / 16.0
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=DATA_SPLIT_SEED,
    )
    return (
        torch.tensor(x_train),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(x_test),
        torch.tensor(y_test, dtype=torch.long),
    )


def train(
    experiment: Experiment,
    seed: int,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[dict[str, list[float]], float]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_train, y_train, x_test, y_test = data
    model = MLP(
        hidden=experiment.hidden,
        activation=experiment.activation,
        dropout=experiment.dropout,
        use_batch_norm=experiment.use_batch_norm,
    )
    if experiment.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=experiment.learning_rate,
            weight_decay=experiment.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=experiment.learning_rate,
            weight_decay=experiment.weight_decay,
        )

    loss_function = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": []}
    start_time = time.perf_counter()

    for _ in range(experiment.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(x_test).argmax(dim=1)
            accuracy = (predictions == y_test).float().mean().item()
        history["loss"].append(loss.item())
        history["accuracy"].append(accuracy)

    return history, time.perf_counter() - start_time


def save_history(filename: str, history: dict[str, list[float]]) -> None:
    with open(f"history_{filename}.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "training_loss", "test_accuracy"])
        for epoch, (loss, accuracy) in enumerate(
            zip(history["loss"], history["accuracy"]), start=1
        ):
            writer.writerow([epoch, loss, accuracy])


def save_curves(experiment: Experiment, history: dict[str, list[float]]) -> None:
    epochs = range(1, experiment.epochs + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["loss"], color="#2878B5")
    axes[0].set_title("训练损失曲线")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, np.array(history["accuracy"]) * 100, color="#C82423")
    axes[1].set_title("测试准确率曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("测试准确率 (%)")
    axes[1].grid(alpha=0.25)

    figure.suptitle(f"组 {experiment.group}：{experiment.name}")
    figure.tight_layout()
    figure.savefig(f"result_{experiment.filename}.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_once(
    experiment: Experiment,
    seed: int,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, str | float]:
    history, elapsed = train(experiment, seed, data)
    save_history(experiment.filename, history)
    save_curves(experiment, history)
    accuracy = history["accuracy"][-1] * 100
    print(
        f"组 {experiment.group} {experiment.name}: "
        f"测试准确率 {accuracy:.2f}%, 训练耗时 {elapsed:.3f}s"
    )
    return {
        "group": experiment.group,
        "name": experiment.name,
        "change": experiment.change,
        "accuracy_percent": accuracy,
        "training_seconds": elapsed,
        "seed": str(seed),
        "notes": "",
    }


def run_best_three_times(
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, str | float]:
    experiment = EXPERIMENTS["best"]
    histories = []
    elapsed_times = []
    final_accuracies = []

    for run_number, seed in enumerate(BEST_MODEL_SEEDS, start=1):
        history, elapsed = train(experiment, seed, data)
        histories.append(history)
        elapsed_times.append(elapsed)
        final_accuracies.append(history["accuracy"][-1] * 100)

        run_experiment = replace(
            experiment,
            name=f"最优组合第 {run_number} 次复测",
            filename=f"exp8_run{run_number}",
        )
        save_history(run_experiment.filename, history)
        save_curves(run_experiment, history)

    mean_history = {
        "loss": np.mean([history["loss"] for history in histories], axis=0).tolist(),
        "accuracy": np.mean(
            [history["accuracy"] for history in histories], axis=0
        ).tolist(),
    }
    save_history("exp8_mean", mean_history)

    mean_accuracy = float(np.mean(final_accuracies))
    accuracy_std = float(np.std(final_accuracies))
    mean_elapsed = float(np.mean(elapsed_times))
    print(
        f"组 8 最优组合（种子 {BEST_MODEL_SEEDS}）: "
        f"平均测试准确率 {mean_accuracy:.2f}% ± {accuracy_std:.2f}%, "
        f"平均训练耗时 {mean_elapsed:.3f}s"
    )
    return {
        "group": experiment.group,
        "name": experiment.name,
        "change": experiment.change,
        "accuracy_percent": mean_accuracy,
        "training_seconds": mean_elapsed,
        "seed": "/".join(map(str, BEST_MODEL_SEEDS)),
        "notes": f"三次准确率标准差 {accuracy_std:.4f}",
    }


def save_results(rows: list[dict[str, str | float]]) -> None:
    with open("results.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison(rows: list[dict[str, str | float]]) -> None:
    improved_rows = [row for row in rows if row["group"] != "失败"]
    labels = [str(row["group"]) for row in improved_rows]
    accuracies = [float(row["accuracy_percent"]) for row in improved_rows]
    elapsed_times = [float(row["training_seconds"]) for row in improved_rows]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    accuracy_bars = axes[0].bar(labels, accuracies, color="#2878B5")
    axes[0].set_title("各组最终测试准确率")
    axes[0].set_xlabel("实验组号")
    axes[0].set_ylabel("测试准确率 (%)")
    axes[0].bar_label(accuracy_bars, fmt="%.1f", padding=2, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    time_bars = axes[1].bar(labels, elapsed_times, color="#F8AC8C")
    axes[1].set_title("各组训练耗时")
    axes[1].set_xlabel("实验组号")
    axes[1].set_ylabel("训练耗时 (s)")
    axes[1].bar_label(time_bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle("基线、单项改进与最优组合结果对比")
    figure.tight_layout()
    figure.savefig("result_summary.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行作业一 MLP 控制变量实验")
    parser.add_argument(
        "--experiment",
        choices=["all", *EXPERIMENTS.keys()],
        default="all",
        help="选择单组实验；all 会依次运行全部实验",
    )
    args = parser.parse_args()
    data = load_data()

    if args.experiment == "all":
        rows = [
            run_once(EXPERIMENTS[key], DATA_SPLIT_SEED, data)
            for key in ("baseline", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7")
        ]
        rows.append(run_once(EXPERIMENTS["failure"], DATA_SPLIT_SEED, data))
        rows.append(run_best_three_times(data))
        save_results(rows)
        save_comparison(rows)
    elif args.experiment == "best":
        save_results([run_best_three_times(data)])
    else:
        save_results(
            [run_once(EXPERIMENTS[args.experiment], DATA_SPLIT_SEED, data)]
        )


if __name__ == "__main__":
    main()
