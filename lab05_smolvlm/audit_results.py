"""核对已保存的真实实验记录，生成精度对比及失败案例；不会训练或改原始预测。"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from hw5_dpo import QUESTIONS, EXTRA_QUESTIONS, IMG_SPEC, judge, guide_judge, write_csv, write_json

STAGES=["baseline","sft","dpo12_gc","dpo17_gc","beta005","beta05"]

def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def verify(out):
    global STAGES
    dpo12_run="dpo12_gc" if (out/"dpo12_gc/history.csv").exists() else "dpo12"
    dpo17_run="dpo17_gc" if (out/"dpo17_gc/history.csv").exists() else "dpo17"
    STAGES=["baseline","sft",dpo12_run,dpo17_run,"beta005","beta05"]
    all_questions=QUESTIONS+EXTRA_QUESTIONS
    expected={q["id"]:q for q in all_questions}
    checks={}
    for q in all_questions:
        color=next(c for c in ["红","蓝","绿","黄"] if c+"色" in q["q"])
        objects=[(s,n) for c,s,n in IMG_SPEC[q["img"]] if c==color]
        if q["type"]=="count":
            shape="圆" if "圆形" in q["q"] else "方"
            gold=str(sum(n for s,n in objects if s==shape))
        elif q["type"]=="attr":
            gold=objects[0][0]+"形"
        else:
            shape="圆" if "圆形" in q["q"] else ("方" if "方形" in q["q"] else None)
            gold="有" if any(n>0 and (shape is None or s==shape) for s,n in objects) else "没有"
        assert gold==q["gold"],q
    checks["all_20_gold_answers_match_image_spec"]=True
    all_metrics=[]
    comparison=[]
    predictions={}
    for precision,folder in [("fp32","evaluations"),("nf4","evaluations_nf4")]:
        for stage in STAGES:
            p=out/folder/stage
            rows=read_json(p/"predictions.json")
            predictions[precision,stage]=rows
            assert len(rows)==20 and {r["id"] for r in rows}==set(expected)
            for row in rows:
                q=expected[row["id"]]
                assert all(row[k]==q[k] for k in ["img","q","gold","type"])
                assert row["correct"]==judge(q,row["answer"])
                assert row["guide_substring_correct"]==guide_judge(q,row["answer"])
            for metric in read_csv(p/"metrics.csv"):
                part=[r for r in rows if r["question_set"]==metric["question_set"] and
                      (metric["type"]=="all" or r["type"]==metric["type"])]
                assert len(part)==int(metric["n"])
                assert sum(r["correct"] for r in part)==int(metric["correct"])
                all_metrics.append(dict(precision=precision,**metric))
            checks[f"{precision}_{stage}_20_predictions_verified"]=True
    for stage in STAGES:
        result=dict(stage=stage)
        for precision in ["fp32","nf4"]:
            for question_set in ["builtin15","extra5"]:
                rows=[r for r in predictions[precision,stage] if r["question_set"]==question_set]
                result[f"{precision}_{question_set}_correct"]=sum(r["correct"] for r in rows)
        comparison.append(result)
    write_csv(out/"all_metrics_nf4.csv",[r for r in all_metrics if r["precision"]=="nf4"])
    write_csv(out/"precision_comparison.csv",comparison)

    histories={}
    training=[]
    for stage in STAGES[1:]:
        config=read_json(out/stage/"config.json")
        history=read_csv(out/stage/"history.csv")
        histories[stage]=history
        steps=60 if stage=="sft" else 40
        assert config["actual_steps"]==steps==len(history)
        assert [int(r["step"]) for r in history]==list(range(1,steps+1))
        assert config["seed"]==42 and config["r"]==8 and config["alpha"]==16
        assert config["trainable"]==1363968 and config["total"]==257848896
        assert read_json(out/stage/"first_step_update.json")["squared_change"]>0
        for row in history:
            assert math.isfinite(float(row["loss"])) and math.isfinite(float(row["grad_norm"]))
            if stage!="sft":
                margin=config["beta"]*((float(row["policy_chosen"])-float(row["reference_chosen"]))-
                                       (float(row["policy_rejected"])-float(row["reference_rejected"])))
                assert abs(margin-float(row["margin"]))<2e-5
                loss=max(-margin,0)+math.log1p(math.exp(-abs(margin)))
                assert abs(loss-float(row["loss"]))<2e-5
                assert all(float(row[k])<=1e-5 for k in ["policy_chosen","policy_rejected",
                                                        "reference_chosen","reference_rejected"])
        n=config["pairs"]
        summary=dict(stage=stage,actual_steps=steps,pairs=n,beta=config["beta"],
            training_s=config["training_s"],total_s=config["total_s"],
            peak_allocated_mib=config["peak_allocated_mib"],
            loss_epoch1=statistics.mean(float(r["loss"]) for r in history[:n]),
            loss_epoch2=statistics.mean(float(r["loss"]) for r in history[n:2*n]),
            margin_epoch1="",margin_epoch2="")
        if stage!="sft":
            summary.update(margin_epoch1=statistics.mean(float(r["margin"]) for r in history[:n]),
                margin_epoch2=statistics.mean(float(r["margin"]) for r in history[n:2*n]))
        training.append(summary)
    for field in ["pair_id","sample_index"]:
        sequences=[[r[field] for r in histories[s]] for s in [dpo17_run,"beta005","beta05"]]
        assert sequences[0]==sequences[1]==sequences[2]
    for stage in [dpo17_run,"beta005","beta05"]:
        config=read_json(out/stage/"config.json")
        assert config["lr"]==5e-5 and config["pairs"]==17
        assert config["dropout"]==0 and config["gradient_checkpointing"] is True
        assert config["starting_adapter"]==read_json(out/dpo17_run/"config.json")["starting_adapter"]
        for key in ["policy_chosen","policy_rejected","reference_chosen","reference_rejected"]:
            assert abs(float(histories[stage][0][key])-float(histories[dpo17_run][0][key]))<1e-4
    checks["beta_order_budget_initial_likelihood_and_adapter_identical"]=True
    checks["all_220_optimizer_steps_and_dpo_formulas_verified"]=True
    failed_count=0
    for failed in ["dpo12","dpo17"]:
        if (out/failed/"training.jsonl").exists() and not (out/failed/"history.csv").exists():
            assert (out/failed/"training.jsonl").read_text()==""
            failed_count+=1
    checks["initial_oom_attempts_zero_optimizer_steps"]=failed_count
    write_csv(out/"training_summary.csv",training)
    write_json(out/"experiment_audit.json",checks)

    failures=[]
    for precision in ["fp32","nf4"]:
        for qid,category,note in [
            ("P01","数错","图中有两个红圆；对照实际输出判断，不能把单独数字当作正确事实。"),
            ("P02","数错","图中有三个绿圆；数数错误不等同于存在性拒答。"),
            ("P11","存在物体误拒答","图中有一个黄方，回答 No 不是诚实拒答。"),
            ("P13","存在物体误拒答","图中有一个绿圆，回答 No 不是诚实拒答。"),
            ("P12","不存在物体硬答有或理由偏离","图中只有红方与蓝方，没有红圆；须区分结论、理由和无意义片段。"),
            ("P09","答案词正确但对象绑定可疑","回答 circle 可命中形状词，但若描述 green outline，则不是问句中的黄色对象；图中确有绿圆，不能说绿色本身是编造。"),
        ]:
            for stage in ["baseline","sft",dpo17_run]:
                row=next(r for r in predictions[precision,stage] if r["id"]==qid)
                failures.append(dict(precision=precision,stage=stage,id=qid,img=row["img"],question=row["q"],
                    gold=row["gold"],answer=row["answer"],correct=row["correct"],
                    candidate_category=category,evidence_note=note,
                    interpretation="候选案例，需结合原文/图像逐条判断；不是所有行都失败"))
    write_csv(out/"failure_case_evidence.csv",failures)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    x=np.arange(len(comparison))
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(x-.2,[r["fp32_builtin15_correct"] for r in comparison],.4,label="FP32 evaluation")
    ax.bar(x+.2,[r["nf4_builtin15_correct"] for r in comparison],.4,label="NF4 evaluation")
    ax.set(xticks=x,xticklabels=[r["stage"] for r in comparison],ylabel="Correct / 15",ylim=(0,16),
           title="Precision diagnostic (same questions, decoding and adapters)")
    ax.tick_params(axis="x",rotation=20);ax.legend();fig.tight_layout()
    fig.savefig(out/"precision_comparison.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for typ,ax in zip(["count","attr","exist"],axes):
        rows=[next(r for r in all_metrics if r["precision"]=="nf4" and r["stage"]==stage and
                   r["question_set"]=="builtin15" and r["type"]==typ) for stage in STAGES]
        ax.bar(STAGES,[float(r["accuracy"])*100 for r in rows])
        ax.set(title=typ,ylabel="NF4 accuracy (%)",ylim=(0,105));ax.tick_params(axis="x",rotation=35)
    fig.tight_layout();fig.savefig(out/"accuracy_nf4.png",dpi=180);plt.close(fig)
    print('AUDIT_PASSED',len(checks),'checks')
    print(json.dumps(comparison,ensure_ascii=False))
    print(json.dumps(training,ensure_ascii=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--outputs",type=Path,required=True)
    verify(ap.parse_args().outputs)
