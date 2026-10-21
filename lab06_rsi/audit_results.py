"""只读取实验输出，核对指标、训练预算、新样本使用和网格对照。"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from hw6_rsi_localize import read_json, write_json, write_csv, metrics, judge, parse_cell


def audit(root, run_name=""):
    outputs = root/"outputs"/run_name
    comparisons = []
    accepted_total, self_attempts_total = 0, 0
    total_steps = 0
    cases, verifier_errors = {}, []
    for grid in [4, 8]:
        out = outputs/f"grid{grid}"
        qs = read_json(out/"questions.json")
        assert len(qs) == 15
        config = read_json(out/"config.json")
        assert config["actual_rounds"] == 2
        seed_steps = config["actual_seed_steps"]
        round_steps = config.get("actual_round_steps", 10)
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
            expected_steps = seed_steps if phase == "sft_train" else round_steps
            assert len(rows) == expected_steps
            assert [int(r["step"]) for r in rows] == list(range(1, expected_steps+1))
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
        assert read_json(out/"completion.json")["optimizer_steps"] == seed_steps+2*round_steps
        total_steps += seed_steps+2*round_steps
        gold_cells = Counter(tuple(q["cell"]) for q in qs if q["type"] == "loc")
        diagnostic = dict(majority_cell_correct=max(gold_cells.values()), loc_n=8,
                          unique_pool_questions=[len({(r["img"], r["prompt"]) for r in pool}) for pool in pools],
                          stages={})
        for stage in ["baseline", "sft_eval", "sft_harness", "rsi1_after", "rsi2_after"]:
            m = read_json(out/stage/"metrics.json")
            comparisons.append(dict(grid=grid, stage=stage, loc_correct=m["loc_correct"],
                count_correct=m["count_correct"], exist_correct=m["exist_correct"],
                correct=m["correct"], loc_mean_iou=m["loc_mean_iou"], loc_parse_success=m["loc_parse_success"]))
            correction = stage == "sft_harness"
            records = read_json(out/stage/("records.json" if correction else "predictions.json"))
            key = "final" if correction else "answer"
            predicted_cells = [parse_cell(r[key], grid) for r in records if r["type"] == "loc"]
            diagnostic["stages"][stage] = dict(
                unique_valid_cells=len({c for c in predicted_cells if c is not None}),
                cell_histogram=dict(Counter(str(c) for c in predicted_cells)),
                ambiguous_count_answers=[dict(id=r["id"], answer=r[key]) for r in records
                    if r["type"] == "count" and len(set(re.findall(r"\d+", r[key]))) > 1])
        write_json(out/"response_diagnostics.json", diagnostic)
    c4 = read_json(outputs/"grid4/config.json")
    c8 = read_json(outputs/"grid8/config.json")
    assert c4["actual_seed_steps"] == c8["actual_seed_steps"]
    assert c4.get("actual_round_steps", 10) == c8.get("actual_round_steps", 10)
    assert c4["model_weights_sha256"] == c8["model_weights_sha256"]
    assert c4["data_manifest_sha256"] == c8["data_manifest_sha256"]
    assert read_json(outputs/"grid4/parameters.json")["initial_adapter_sha256"] == read_json(outputs/"grid8/parameters.json")["initial_adapter_sha256"]
    q4 = read_json(outputs/"grid4/questions.json")
    q8 = read_json(outputs/"grid8/questions.json")
    assert [(q["id"], q["img"], q["target"], q["type"]) for q in q4] == [(q["id"], q["img"], q["target"], q["type"]) for q in q8]
    write_csv(outputs/"grid_comparison.csv", comparisons)
    for grid in [4, 8]:
        for kind in ["direct_correct", "self_fixed", "tool_fixed", "still_wrong"]:
            cases.setdefault(f"grid{grid}_{kind}", dict(status="not_observed"))
    write_json(outputs/"representative_cases.json", cases)
    write_json(outputs/"verifier_disagreements.json", verifier_errors)
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
    fig.savefig(outputs/"grid_comparison.png", dpi=180)
    plt.close(fig)
    write_json(outputs/"audit.json", dict(status="passed", grids=[4, 8],
        total_optimizer_steps=total_steps, identical_model_and_scenes=True,
        metric_recalculation=True, all_accepted_samples_used=True,
        accepted_total=accepted_total, bootstrap_observed=accepted_total > 0,
        self_check_attempts=self_attempts_total, self_check_observed=self_attempts_total > 0))
    print(f"AUDIT_PASSED grids=4/8 steps={total_steps} metrics pools transitions fairness")


def compare_runs(root, names):
    """并列保留每个预算，不以追加结果覆盖较差的原始尝试。"""
    rows = []
    for name in names:
        base = root/"outputs"/("" if name == "baseline10" else name)
        for grid in [4, 8]:
            out = base/f"grid{grid}"
            config = read_json(out/"config.json")
            rounds = list(csv.DictReader((out/"round_metrics.csv").open(encoding="utf-8-sig")))
            for stage in ["baseline", "sft_eval", "sft_harness", "rsi1_after", "rsi2_after"]:
                m = read_json(out/stage/"metrics.json")
                rows.append(dict(run=name, grid=grid, seed_steps=config["actual_seed_steps"],
                    rsi_steps=config.get("actual_round_steps", 10), stage=stage,
                    loc_correct=m["loc_correct"], loc_parse=m["loc_parse_success"],
                    correct=m["correct"], loc_mean_iou=m["loc_mean_iou"],
                    final_pool=int(rounds[-1]["pool_size"]),
                    total_accepted=sum(int(r["new_accepted"]) for r in rounds)))
    write_csv(root/"outputs/extension_budget_comparison.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for grid, ax in zip([4, 8], axes):
        for phase in ["sft_eval", "rsi2_after"]:
            subset = [r for r in rows if r["grid"] == grid and r["stage"] == phase]
            ax.plot([f"S{r['seed_steps']}/R{r['rsi_steps']}" for r in subset], [r["loc_correct"]/8 for r in subset], marker="o", label=phase)
        ax.set(title=f"Grid {grid}: budget comparison", xlabel="Seed / per-round RSI steps", ylabel="Localization accuracy", ylim=(-.03, 1.03))
        ax.legend()
        ax.grid(alpha=.25)
    fig.savefig(root/"outputs/extension_budget_comparison.png", dpi=180)
    plt.close(fig)
    print("BUDGET_COMPARISON_SAVED", names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--compare-runs", nargs="+")
    args = parser.parse_args()
    if args.compare_runs:
        compare_runs(args.root, args.compare_runs)
    else:
        audit(args.root, args.run_name)
