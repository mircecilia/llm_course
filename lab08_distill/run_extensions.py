# -*- coding: utf-8 -*-
"""补跑拓展 A、C、D、F，复用已完成的 B、E 和主对照。"""

import argparse
import ast
import hashlib
import io
import json
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from distill_lm import (ROOT, STUDENT_CFG, TEACHER_CFG, Tokenizer, evaluate,
                        get_batch, make_data, metric, new_model, show, train,
                        train_student, train_teacher, write_json)


def load_state(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def save_state(model, path):
    # 两种精度使用相同的 ZIP 根目录名，便于直接比较体积。
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    path.write_bytes(buffer.getbuffer())
    return path.stat().st_size


def fitnet(out, args, tokenizer, data, val, main):
    teacher = load_state(new_model(tokenizer, TEACHER_CFG, args.seed), out.parent / "teacher.pt")
    model, fit = train_student("fitnet", tokenizer, data, val, args, teacher, kd=True, feat=0.1)
    baseline = main["kd"]
    assert fit["initial_sha256"] == baseline["initial_sha256"]
    assert fit["batch_seed"] == baseline["batch_seed"]
    save_state(model, out / "student_fitnet.pt")
    return dict(baseline=baseline, fitnet=fit, feature_weight=0.1,
                projection=[STUDENT_CFG["n_embd"], TEACHER_CFG["n_embd"]],
                delta_ce=fit["val_loss"] - baseline["val_loss"])


def self_distill(out, args, tokenizer, data, val, main):
    teacher = new_model(tokenizer, STUDENT_CFG, args.seed)
    steps = 3 * args.student_iters
    curve, elapsed = train(teacher, data, steps, args.batch_size, STUDENT_CFG["block_size"],
                           lr=args.lr, seed=args.seed, log_every=args.log_every, tag="self-teacher")
    reference = metric(teacher, val, args, curve, elapsed)
    show("self-teacher", reference)
    student, distilled = train_student("self-kd", tokenizer, data, val, args, teacher, kd=True)
    baseline = main["scratch"]
    assert distilled["initial_sha256"] == baseline["initial_sha256"]
    assert distilled["batch_seed"] == baseline["batch_seed"]
    save_state(teacher, out / "self_teacher.pt")
    save_state(student, out / "student_self_kd.pt")
    return dict(teacher=reference, baseline=baseline, self_kd=distilled,
                delta_ce=distilled["val_loss"] - baseline["val_loss"])


@torch.inference_mode()
def latency(models, x, warmup=20, repeats=60):
    samples = [[], []]
    for i in range(warmup + repeats):
        for j in (i % 2, 1 - i % 2):
            started = time.perf_counter_ns()
            models[j](x)
            duration = (time.perf_counter_ns() - started) / 1e6
            if i >= warmup:
                samples[j].append(duration)
    return dict(warmup=warmup, repeats=repeats, input_shape=list(x.shape),
                fp32_ms=statistics.median(samples[0]), int8_ms=statistics.median(samples[1]),
                samples_ms=samples)


def quantize(out, args, tokenizer, data, val, main):
    model = load_state(new_model(tokenizer, STUDENT_CFG, args.seed + 7),
                       out.parent / "student_kd.pt")
    quant = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    count = sum(isinstance(layer, torch.ao.nn.quantized.dynamic.Linear) for layer in quant.modules())
    assert count == sum(isinstance(layer, nn.Linear) for layer in model.modules())
    fp_path, q_path = out / "student_fp32.pt", out / "student_int8.pt"
    fp_bytes, q_bytes = save_state(model, fp_path), save_state(quant, q_path)
    # 以导出后重新载入的权重做质量和延迟评测。
    reloaded = torch.ao.quantization.quantize_dynamic(
        new_model(tokenizer, STUDENT_CFG, args.seed + 7).eval(), {nn.Linear}, dtype=torch.qint8)
    load_state(reloaded, q_path)
    load_state(model, fp_path)
    block = STUDENT_CFG["block_size"]
    fp_ce, q_ce = (evaluate(m, val, args.batch_size, block) for m in (model, reloaded))
    assert abs(fp_ce - main["kd"]["val_loss"]) < 1e-6
    x, _ = get_batch(val, 1, block, torch.Generator().manual_seed(1234))
    bench = latency((model.eval(), reloaded.eval()), x)
    return dict(engine=torch.backends.quantized.engine, threads=torch.get_num_threads(),
                quantized_linear_layers=count, fp32_bytes=fp_bytes, int8_bytes=q_bytes,
                size_ratio=fp_bytes / q_bytes, fp32_ce=fp_ce, int8_ce=q_ce,
                delta_ce=q_ce - fp_ce, **bench)


def load_poems(path):
    # 仅提取实验三的字面量，不导入该脚本及其额外依赖。
    module = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(node for node in module.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "POEMS" for t in node.targets))
    poems = ast.literal_eval(assignment.value)
    assert len(poems) == len(set(poems))
    return poems


def real_corpus(out, args, source, main):
    poems = load_poems(source)
    # 实验三把部分长诗拆成连续条目。明显同篇或高度重合的条目同组划分。
    related = [(4, 5), (19, 20), (30, 31), (32, 33), (39, 40), (54, 55)]
    group_id = {right: left for left, right in related}
    groups = list(dict.fromkeys(group_id.get(i, i) for i in range(len(poems))))
    random.Random(999).shuffle(groups)
    held = set(groups[:max(1, len(groups) // 5)])
    train_idx = [i for i in range(len(poems)) if group_id.get(i, i) not in held]
    val_idx = [i for i in range(len(poems)) if group_id.get(i, i) in held]
    train_text, val_text = ("\n".join(poems[i] for i in idx) for idx in (train_idx, val_idx))
    unk = "\ufffd"
    tokenizer = Tokenizer(train_text + unk)
    unknown = tokenizer.stoi[unk]
    data, val = (torch.tensor([tokenizer.stoi.get(c, unknown) for c in text], dtype=torch.long)
                 for text in (train_text, val_text))
    corpus = dict(source=source.name, poems_sha256=hashlib.sha256("\n".join(poems).encode()).hexdigest(),
                  split_seed=999, train_indices=train_idx, val_indices=val_idx, vocab=tokenizer.vocab_size,
                  train_chars=len(data), val_chars=len(val), val_unk=int((val == unknown).sum()))
    print("[real corpus]", json.dumps(corpus), flush=True)
    teacher, reference = train_teacher(tokenizer, data, val, args, TEACHER_CFG["n_embd"])
    _, scratch = train_student("real-scratch", tokenizer, data, val, args, teacher)
    student, kd = train_student("real-kd", tokenizer, data, val, args, teacher, kd=True)
    assert scratch["initial_sha256"] == kd["initial_sha256"] and scratch["batch_seed"] == kd["batch_seed"]
    save_state(teacher, out / "real_teacher.pt")
    save_state(student, out / "real_student_kd.pt")
    return dict(corpus=corpus, teacher=reference, scratch=scratch, kd=kd,
                delta_ce=kd["val_loss"] - scratch["val_loss"],
                synthetic_delta_ce=main["kd"]["val_loss"] - main["scratch"]["val_loss"])


def kl_check():
    gen = torch.Generator().manual_seed(8)
    student, teacher = torch.randn(2, 96, 16, generator=gen).double(), torch.randn(2, 96, 16, generator=gen).double()
    log_p, q = F.log_softmax(student / 4, dim=-1), F.softmax(teacher / 4, dim=-1)
    correct = F.kl_div(log_p.flatten(0, 1), q.flatten(0, 1), reduction="batchmean").item()
    unflattened = F.kl_div(log_p, q, reduction="batchmean").item()
    assert abs(unflattened / correct - 96) < 1e-8
    return dict(batch=2, sequence=96, vocab=16, flattened=correct, unflattened=unflattened,
                ratio=unflattened / correct)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs_distill",
                        help="包含主对照结果及其检查点的目录")
    parser.add_argument("--corpus-source", type=Path, default=ROOT.parent / "lab03_min_llm" / "min_llm.py")
    cli = parser.parse_args()
    out = cli.output / "extensions"
    out.mkdir(exist_ok=True)
    main_result = json.loads((cli.output / "results_all.json").read_text(encoding="utf-8"))
    protocol = main_result["protocol"]
    torch.set_num_threads(protocol["threads"])
    args = SimpleNamespace(**{k: protocol[k] for k in ("seed", "lr", "T", "alpha")},
                           batch_size=protocol["batch"], student_iters=protocol["student_steps"],
                           teacher_iters=protocol["teacher_steps"], log_every=40)
    tok, data, val = make_data(STUDENT_CFG["block_size"])
    assert (len(data), len(val), tok.vocab_size) == (protocol["train_chars"], protocol["val_chars"], protocol["vocab"])
    stages = {"guide_kl_check": kl_check,
              "D_quantize": lambda: quantize(out, args, tok, data, val, main_result),
              "A_fitnet": lambda: fitnet(out, args, tok, data, val, main_result),
              "C_self_distill": lambda: self_distill(out, args, tok, data, val, main_result),
              "F_real_corpus": lambda: real_corpus(out, args, cli.corpus_source, main_result)}
    for name, run in stages.items():
        target = out / f"{name}.json"
        if target.is_file():
            print(f"[skip] {target.name} already exists", flush=True)
            continue
        started = time.perf_counter()
        result = run()
        result["wall_s"] = time.perf_counter() - started
        write_json(target, result)
        print(f"[done] {target.name} {result['wall_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
