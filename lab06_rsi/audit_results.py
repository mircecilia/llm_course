"""只读取实验输出，核对指标、训练预算、新样本使用和网格对照。"""
import argparse
import csv
import json
from pathlib import Path
from hw6_rsi_localize import read_json, write_json, write_csv, metrics, judge, parse_cell


def audit(root):
    comparisons = []
    accepted_total, self_attempts_total = 0, 0
    cases, verifier_errors = {}, []
    for grid in [4, 8]:
        out = root/"outputs"/f"grid{grid}"
        qs = read_json(out/"questions.json")
        assert len(qs) == 15
        config = read_json(out/"config.json")
        assert config["actual_seed_steps"] == 10 and config["actual_rounds"] == 2
        for stage in ["baseline", "sft_eval", "rsi1_before", "rsi1_after", "rsi2_before", "rsi2_after"]:
            records = read_json(out/stage/"predictions.json")
            recorded = read_json(out/stage/"metrics.json")
            assert len(records) == 15
            assert all(r["id"] == q["id"] and r["q"] == q["q"] for q, r in zip(qs, records))
            for key, val in metrics(qs, records, grid).items():
                assert recorded[key] == val, (grid, stage, key)
        assert [r["answer"] for r in read_json(out/"sft_eval"/"predictions.json")] == [r["answer"] for r in read_json(out/"rsi1_before"/"predictions.json")]
        assert [r["answer"] for r in read_json(out/"rsi1_after"/"predictions.json")] == [r["answer"] for r in read_json(out/"rsi2_before"/"predictions.json")]
        for stage in ["sft_harness", "rsi1_correction", "rsi2_correction"]:
            records = read_json(out/stage/"records.json")
            recorded = read_json(out/stage/"metrics.json")
            assert len(records) == 15
            for key, val in metrics(qs, records, grid, "final").items():
                assert recorded[key] == val
            assert recorded["wrong_to_right"]-recorded["right_to_wrong"] == recorded["correct"]-recorded["direct_correct"]
            for r in records:
                if r["ok1"]:
                    cases.setdefault(f"grid{grid}_direct_correct", dict(stage=stage, **r))
                if r["pred2_self"] is not None and judge(r, r["pred2_self"], grid):
                    cases.setdefault(f"grid{grid}_self_fixed", dict(stage=stage, **r))
                if not r["ok1"] and r["pred2_tool"] is not None and r["ok_final"]:
                    cases.setdefault(f"grid{grid}_tool_fixed", dict(stage=stage, **r))
                if not r["ok_final"]:
                    cases.setdefault(f"grid{grid}_still_wrong", dict(stage=stage, **r))
                if r.get("verifier_agrees_before") is False or r.get("verifier_agrees_final") is False:
                    verifier_errors.append(dict(grid=grid, stage=stage, **r))
                if r["accepted"]:
                    assert r["ok_final"] and r["verifier_final"]["passed"]
                    assert r["final"] in [r["pred2_self"], r["pred2_tool"]]
        for phase in ["sft_train", "rsi1_train", "rsi2_train"]:
            rows = list(csv.DictReader((out/phase/"history.csv").open(encoding="utf-8-sig")))
            assert len(rows) == 10
            assert [int(r["step"]) for r in rows] == list(range(1, 11))
            trained = read_json(out/phase/"training.json")
            # LoRA B 初始为零；保存值非零可独立确认确实发生了参数更新。
            from safetensors.numpy import load_file
            weights = load_file(out/phase/"adapter"/"adapter_model.safetensors")
            assert any((value != 0).any() for name, value in weights.items() if "lora_B" in name)
            if phase != "sft_train":
                assert trained["new_unique_samples_used"] == trained["new_samples_available"]
        pools = [read_json(out/f"pool_round{i}.json") for i in [0, 1, 2]]
        assert len(pools[0]) == 15
        for i in [1, 2]:
            accepted = read_json(out/f"rsi{i}_correction"/"accepted.json")
            accepted_total += len(accepted)
            self_attempts_total += read_json(out/f"rsi{i}_correction"/"metrics.json")["self_attempts"]
            assert pools[i] == pools[i-1]+accepted
        assert read_json(out/"completion.json")["optimizer_steps"] == 30
        for stage in ["baseline", "sft_eval", "sft_harness", "rsi1_after", "rsi2_after"]:
            m = read_json(out/stage/"metrics.json")
            comparisons.append(dict(grid=grid, stage=stage, loc_correct=m["loc_correct"],
                count_correct=m["count_correct"], exist_correct=m["exist_correct"],
                correct=m["correct"], loc_mean_iou=m["loc_mean_iou"], loc_parse_success=m["loc_parse_success"]))
    c4 = read_json(root/"outputs/grid4/config.json")
    c8 = read_json(root/"outputs/grid8/config.json")
    assert c4["model_weights_sha256"] == c8["model_weights_sha256"]
    assert c4["data_manifest_sha256"] == c8["data_manifest_sha256"]
    assert read_json(root/"outputs/grid4/parameters.json")["initial_adapter_sha256"] == read_json(root/"outputs/grid8/parameters.json")["initial_adapter_sha256"]
    q4 = read_json(root/"outputs/grid4/questions.json")
    q8 = read_json(root/"outputs/grid8/questions.json")
    assert [(q["id"], q["img"], q["target"], q["type"]) for q in q4] == [(q["id"], q["img"], q["target"], q["type"]) for q in q8]
    write_csv(root/"outputs/grid_comparison.csv", comparisons)
    for grid in [4, 8]:
        for kind in ["direct_correct", "self_fixed", "tool_fixed", "still_wrong"]:
            cases.setdefault(f"grid{grid}_{kind}", dict(status="not_observed"))
    write_json(root/"outputs/representative_cases.json", cases)
    write_json(root/"outputs/verifier_disagreements.json", verifier_errors)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    labels = ["Base", "SFT", "Harness", "RSI1", "RSI2"]
    for grid in [4, 8]:
        rows = [r for r in comparisons if r["grid"] == grid]
        axes[0].plot(labels, [r["loc_correct"]/8 for r in rows], marker="o", label=f"{grid}x{grid}")
        axes[1].plot(labels, [r["loc_mean_iou"] for r in rows], marker="o", label=f"{grid}x{grid}")
    for ax, title in zip(axes, ["Localization accuracy", "Localization mean IoU"]):
        ax.set(title=title, ylim=(-.02, 1.02))
        ax.legend()
        ax.grid(alpha=.25)
    fig.savefig(root/"outputs/grid_comparison.png", dpi=180)
    plt.close(fig)
    write_json(root/"outputs/audit.json", dict(status="passed", grids=[4, 8],
        total_optimizer_steps=60, identical_model_and_scenes=True,
        metric_recalculation=True, all_accepted_samples_used=True,
        accepted_total=accepted_total, bootstrap_observed=accepted_total > 0,
        self_check_attempts=self_attempts_total, self_check_observed=self_attempts_total > 0))
    print("AUDIT_PASSED grids=4/8 steps=60 metrics pools transitions fairness")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    audit(parser.parse_args().root)
