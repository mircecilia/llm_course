# -*- coding: utf-8 -*-
"""实验六：指南同题基准、两轮 RSI，以及独立的 8×8 网格选做。

原始预测、两路纠错、验证器判断和入池过程逐项落盘，不用 GT 替写模型答案。
FP32 基座 + LoRA 与指南一致；虽然指南称 QLoRA，本实现不做 4-bit 量化。
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time

COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220),
          "绿": (40, 170, 70), "黄": (230, 190, 40)}
IMG = 256
SEED = 42


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_sha(model):
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if param.requires_grad:
            digest.update(name.encode())
            digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def draw_scene(rnd):
    items = []
    for _ in range(rnd.randint(1, 4)):
        color = rnd.choice(list(COLORS))
        shape = rnd.choice(["圆", "方"])
        size = rnd.randint(34, 54)
        x, y = rnd.randint(8, IMG-size-8), rnd.randint(8, IMG-size-8)
        items.append((color+shape, [x, y, x+size, y+size]))
    return items


def make_dataset(root):
    from PIL import Image, ImageDraw
    started = time.perf_counter()
    folder = root/"hw6_imgs"
    assert not (folder/"gt.json").exists(), "数据已经存在，不覆盖"
    gt = {}
    for split, n, seed, prefix in [("train", 40, 2026, "t"), ("eval", 15, 42, "e")]:
        (folder/split).mkdir(parents=True, exist_ok=True)
        rnd = random.Random(seed)
        for i in range(n):
            items = draw_scene(rnd)
            im = Image.new("RGB", (IMG, IMG), "white")
            draw = ImageDraw.Draw(im)
            for cls, box in items:
                fn = draw.ellipse if cls[1] == "圆" else draw.rectangle
                fn(box, outline=COLORS[cls[0]], width=4)
            key = f"{prefix}{i:02d}"
            im.save(folder/split/(key+".png"))
            gt[key] = items
    write_json(folder/"gt.json", gt)
    files = sorted(folder.glob("*/*.png")) + [folder/"gt.json"]
    write_json(folder/"manifest.json", {str(p.relative_to(folder)): sha_file(p) for p in files})
    write_json(folder/"timing.json", dict(seconds=time.perf_counter()-started))
    print("DATA_COMPLETE train=40 eval=15 image_size=256 seeds=2026/42", flush=True)
    return gt


def cell_of(box, grid):
    cell = IMG//grid
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    return min(grid-1, int(cy//cell)), min(grid-1, int(cx//cell))


def cell_box(row, col, grid):
    cell = IMG//grid
    return [col*cell, row*cell, (col+1)*cell, (row+1)*cell]


def cell_text(row, col):
    return f"第{row+1}行第{col+1}列"


def parse_cell(text, grid):
    patterns = [r"第\s*(\d+)\s*行\s*第\s*(\d+)\s*列",
                r"r\s*(\d+)\s*c\s*(\d+)", r"(\d+)\s*[,\-，、]\s*(\d+)"]
    for pattern in patterns:
        m = re.search(pattern, text.strip(), flags=re.I)
        if m:
            r, c = map(int, m.groups())
            return (r-1, c-1) if 1 <= r <= grid and 1 <= c <= grid else None
    return None


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/union if union > 0 else 0.0


def name_of(cls):
    return f"{cls[0]}色{cls[1:]}形"


def build_questions(gt, grid):
    qs = []
    for i in range(15):
        eid = f"e{i:02d}"
        items = gt[eid]
        present = sorted({cls for cls, _ in items})
        cls = present[i % len(present)]
        box = next(b for c, b in items if c == cls)
        row, col = cell_of(box, grid)
        qs.append(dict(id=f"L{i+1:02d}", type="loc", img=eid, target=cls,
                       q=f"图中的{name_of(cls)}位于第几行第几列？请用「第 X 行第 Y 列」回答（共{grid}行{grid}列）。",
                       gold=cell_text(row, col), cell=[row, col], box=box))
        n = sum(c == cls for c, _ in items)
        qs.append(dict(id=f"C{i+1:02d}", type="count", img=eid, target=cls,
                       q=f"图中有几个{name_of(cls)}？", gold=f"{n}个", n=n))
        qs.append(dict(id=f"E{i+1:02d}", type="exist", img=eid, target=cls,
                       q=f"图里有没有{name_of(cls)}？", gold="有"))
    return ([q for q in qs if q["type"] == "loc"][:8]
            + [q for q in qs if q["type"] == "count"][:4]
            + [q for q in qs if q["type"] == "exist"][:3])


def judge(q, pred, grid):
    """主指标保持指南规则：固定题集存在题全为正例，不额外补题改分母。"""
    p = pred.strip().lower()
    if q["type"] == "loc":
        return parse_cell(p, grid) == tuple(q["cell"])
    if q["type"] == "count":
        nums = re.findall(r"\d+", p)
        return bool(nums) and int(nums[0]) == q["n"]
    yes = ("有" in p and "没有" not in p.replace("有没有", "")) or "yes" in p
    no = ("没有" in p.replace("有没有", "")) or "no" in p or "not" in p
    return bool(yes if q["gold"] == "有" else no)


def loc_iou(q, pred, grid):
    cell = parse_cell(pred, grid)
    return iou(cell_box(*cell, grid), q["box"]) if cell else 0.0


def image(root, q):
    from PIL import Image
    return Image.open(root/"hw6_imgs"/"eval"/(q["img"]+".png")).convert("RGB")


def color_components(img, color):
    px = img.convert("RGB").load()
    w, h = img.size
    ref = COLORS[color]
    points = {(x, y) for y in range(h) for x in range(w)
              if sum((px[x, y][i]-ref[i])**2 for i in range(3)) < 9000}
    components = []
    while points:
        seed = min(points)  # 固定遍历顺序，使诊断记录也能稳定复现。
        points.remove(seed)
        stack, pixels = [seed], [seed]
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = (x+dx, y+dy)
                    if nxt in points:
                        points.remove(nxt)
                        stack.append(nxt)
                        pixels.append(nxt)
        xs, ys = zip(*pixels)
        components.append(([min(xs), min(ys), max(xs)+1, max(ys)+1], len(pixels)))
    return components


def verify_cell(img, q, cell, grid):
    """保留参考实现的颜色工具，不偷偷增加形状识别或 GT 坐标提示。"""
    started = time.perf_counter()
    comps = color_components(img, q["target"][0])
    cb = cell_box(*cell, grid)
    passed, best_overlap = False, 0.0
    for box, _ in comps:
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        passed |= cb[0] <= cx < cb[2] and cb[1] <= cy < cb[3]
        inter = max(0, min(cb[2], box[2])-max(cb[0], box[0])) * max(0, min(cb[3], box[3])-max(cb[1], box[1]))
        area = (box[2]-box[0])*(box[3]-box[1])
        best_overlap = max(best_overlap, inter/area if area else 0.0)
    if not comps:
        msg = "该颜色在图中不存在"
    elif passed:
        msg = "高亮格确实定位到了目标（目标中心在该格内）"
    elif best_overlap >= 0.5:
        passed = True
        msg = f"高亮格覆盖了目标 {best_overlap:.0%} 的面积"
    else:
        msg = f"高亮格没有定位到目标（重叠仅 {best_overlap:.0%}），请重新给出位置"
    return bool(passed), msg, dict(components=comps, overlap=best_overlap,
                                  seconds=time.perf_counter()-started)


def color_mask_stats(img, color):
    comps = color_components(img, color)
    return dict(components=len(comps), total=sum(n for _, n in comps))


def overlay(img, cell, grid, path):
    from PIL import ImageDraw
    im = img.copy()
    draw = ImageDraw.Draw(im)
    cb = cell_box(*cell, grid)
    draw.rectangle(cb, outline=(255, 0, 255), width=5)
    draw.rectangle([cb[0]+6, cb[1]+6, cb[2]-6, cb[3]-6], outline=(255, 0, 255), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return im


def seed_data(qs):
    return [dict(img=q["img"], prompt=q["q"], answer=q["gold"],
                 question_id=q["id"], source="seed_gt", origin_round=0) for q in qs]


def set_seed():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)


def load_model(args, adapter=None):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model, PeftModel
    set_seed()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(args.model, torch_dtype=torch.float32,
              local_files_only=True, attn_implementation="eager")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    return processor, model


def attach_lora(model):
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
    return model


def prompt_text(processor, question):
    return processor.apply_chat_template([{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}]}], add_generation_prompt=True)


def answer(processor, model, img, question, hint=None, max_new=24):
    import torch
    model.eval()
    text = question if hint is None else f"{question}\n（校验提示：{hint}）"
    inputs = processor(text=[prompt_text(processor, text)], images=[img],
                       return_tensors="pt").to(next(model.parameters()).device)
    started = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    generated = out[0, inputs["input_ids"].shape[1]:].tolist()
    return processor.decode(generated, skip_special_tokens=True).strip(), dict(
        token_ids=generated, seconds=time.perf_counter()-started, prompt=text)


def metrics(qs, records, grid, key="answer"):
    result = {}
    for tp in ["loc", "count", "exist"]:
        subset = [(q, rec) for q, rec in zip(qs, records) if q["type"] == tp]
        result[tp+"_correct"] = sum(judge(q, rec[key], grid) for q, rec in subset)
        result[tp+"_n"] = len(subset)
    result["correct"] = sum(result[t+"_correct"] for t in ["loc", "count", "exist"])
    result["n"] = len(qs)
    result["accuracy"] = result["correct"]/result["n"]
    locs = [(q, r) for q, r in zip(qs, records) if q["type"] == "loc"]
    result["loc_mean_iou"] = sum(loc_iou(q, r[key], grid) for q, r in locs)/len(locs)
    result["loc_parse_success"] = sum(parse_cell(r[key], grid) is not None for _, r in locs)
    return result


def evaluate(args, processor, model, qs, stage):
    folder = args.out/stage
    folder.mkdir(exist_ok=False)
    records = []
    started = time.perf_counter()
    for q in qs:
        pred, generation = answer(processor, model, image(args.root, q), q["q"])
        rec = dict(**q, answer=pred, correct=judge(q, pred, args.grid), generation=generation)
        if q["type"] == "loc":
            rec["iou"] = loc_iou(q, pred, args.grid)
        records.append(rec)
        write_json(folder/"predictions.json", records)
        print(f"[{stage} {q['id']}] A={pred!r} gold={q['gold']} correct={rec['correct']}", flush=True)
    summary = dict(stage=stage, grid=args.grid, **metrics(qs, records, args.grid),
                   seconds=time.perf_counter()-started)
    write_json(folder/"metrics.json", summary)
    print("EVAL_COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)
    return records, summary


def correct(args, processor, model, qs, direct, stage, self_check=False, round_idx=0):
    folder = args.out/stage
    folder.mkdir(exist_ok=False)
    records, accepted = [], []
    stats = dict(self_attempts=0, self_rewrites=0, tool_calls=0, triggers=0,
                 wrong_to_right=0, right_to_wrong=0, self_fixed=0, tool_fixed=0)
    for q, original in zip(qs, direct):
        im = image(args.root, q)
        pred1 = original["answer"]
        ok1 = judge(q, pred1, args.grid)
        rec = dict(**q, pred1=pred1, ok1=ok1, pred2_self=None, pred2_tool=None,
                   final=pred1, path="direct", round=round_idx, oracle_trigger=not ok1)
        # 指南 RSI 只对定位题纠错；三阶段 Harness 对三类错题均提供反馈。
        if not ok1 and (q["type"] == "loc" or not self_check):
            stats["triggers"] += 1
            current = pred1
            if q["type"] == "loc":
                first_cell = parse_cell(pred1, args.grid)
                if self_check and first_cell is not None:
                    stats["self_attempts"] += 1
                    ov = overlay(im, first_cell, args.grid, folder/"overlays"/(q["id"]+".png"))
                    rec["overlay"] = str((folder/"overlays"/(q["id"]+".png")).relative_to(args.out))
                    current, gen = answer(processor, model, ov,
                        f"图中紫色高亮框是上一次给出的位置。{q['q']}\n若高亮框没有圈住目标，请直接给出正确位置。",
                        max_new=16)
                    rec.update(pred2_self=current, self_generation=gen)
                    stats["self_rewrites"] += parse_cell(current, args.grid) != first_cell
                    stats["self_fixed"] += judge(q, current, args.grid)
                    rec["path"] = "self"
                cell = parse_cell(current, args.grid)
                passed, msg, diagnostic = verify_cell(im, q, cell or (0, 0), args.grid)
                rec.update(verifier_before=dict(passed=passed, message=msg, cell=cell, **diagnostic),
                           verifier_agrees_before=(passed == judge(q, current, args.grid)))
                needs_tool = not passed or cell is None
                if cell is None:
                    msg = "位置格式无法解析，请按第 X 行第 Y 列回答；" + msg
            else:
                st = color_mask_stats(im, q["target"][0])
                rec["color_tool"] = st
                msg = f"图中该颜色共 {st['components']} 处" if q["type"] == "count" else f"图中该颜色像素数 {st['total']}"
                needs_tool = True
            if needs_tool:
                stats["tool_calls"] += 1
                current, gen = answer(processor, model, im, q["q"], hint=msg)
                rec.update(pred2_tool=current, hint=msg, tool_generation=gen, path="tool")
                stats["tool_fixed"] += judge(q, current, args.grid)
            rec["final"] = current
        rec["ok_final"] = judge(q, rec["final"], args.grid)
        stats["wrong_to_right"] += not ok1 and rec["ok_final"]
        stats["right_to_wrong"] += ok1 and not rec["ok_final"]
        rec["accepted"] = False
        if self_check and q["type"] == "loc" and not ok1:
            final_cell = parse_cell(rec["final"], args.grid)
            if final_cell is not None:
                passed, msg, diagnostic = verify_cell(im, q, final_cell, args.grid)
                rec["verifier_final"] = dict(passed=passed, message=msg, **diagnostic)
                rec["accepted"] = bool(passed and rec["ok_final"])
                rec["verifier_agrees_final"] = passed == rec["ok_final"]
            if rec["accepted"]:
                accepted.append(dict(img=q["img"], prompt=q["q"], answer=rec["final"],
                      question_id=q["id"], source="model_verified_correction", origin_round=round_idx,
                      correction_path=rec["path"]))
        records.append(rec)
        write_json(folder/"records.json", records)
        print(f"[{stage} {q['id']}] {pred1!r} -> self={rec['pred2_self']!r} -> tool={rec['pred2_tool']!r} final={rec['final']!r}", flush=True)
    summary = dict(stage=stage, grid=args.grid, **metrics(qs, records, args.grid, key="final"),
                   **stats, accepted=len(accepted), direct_correct=sum(r["ok1"] for r in records))
    write_json(folder/"metrics.json", summary)
    write_json(folder/"accepted.json", accepted)
    print("CORRECTION_COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)
    return records, accepted, summary


def encode(processor, img, prompt, response):
    import torch
    pref = prompt_text(processor, prompt)
    head = processor(text=[pref], images=[img], return_tensors="pt")
    full = processor(text=[pref+response+processor.tokenizer.eos_token], images=[img], return_tensors="pt")
    length = head["input_ids"].shape[1]
    assert torch.equal(head["input_ids"], full["input_ids"][:, :length]), "prompt token 前缀发生变化"
    labels = full["input_ids"].clone()
    labels[:, :length] = -100
    assert (labels != -100).any()
    full["labels"] = labels
    return full, length


def train_phase(args, processor, model, pool, phase, optimizer, new_count=0, round_idx=0):
    import torch
    folder = args.out/phase
    folder.mkdir(exist_ok=False)
    if round_idx:
        new_indices = list(range(len(pool)-new_count, len(pool))) if new_count else []
        old_indices = list(range(len(pool)-new_count))
        random.Random(SEED+round_idx).shuffle(old_indices)
        order = new_indices+old_indices
    else:
        order = list(range(len(pool)))  # 初始 10 步按指南顺序：8 loc + 2 count。
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    history = []
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    started = time.perf_counter()
    for step in range(args.steps):
        idx = order[step % len(order)]
        sample = pool[idx]
        q = {"img": sample["img"]}
        batch, prompt_len = encode(processor, image(args.root, q), sample["prompt"], sample["answer"])
        batch = batch.to(next(model.parameters()).device)
        optimizer.zero_grad()
        loss = model(**batch).loss
        assert torch.isfinite(loss), "loss 非有限值，保留当前记录并停止"
        loss.backward()
        optimizer.step()
        rec = dict(step=step+1, loss=float(loss.detach()), pool_index=idx,
                   question_id=sample["question_id"], source=sample["source"],
                   origin_round=sample["origin_round"], prompt_len=prompt_len,
                   supervised_tokens=int(batch["labels"].ne(-100).sum()))
        history.append(rec)
        write_csv(folder/"history.csv", history)
        print(f"[{phase}] step={step+1}/{args.steps} loss={rec['loss']:.6f} pool_index={idx} source={sample['source']}", flush=True)
    model.save_pretrained(folder/"adapter")
    elapsed = time.perf_counter()-started
    summary = dict(phase=phase, steps=len(history), seconds=elapsed, pool_size=len(pool),
                   new_samples_available=new_count,
                   new_unique_samples_used=len({r["pool_index"] for r in history if r["origin_round"] == round_idx and round_idx > 0}),
                   peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0)
    write_json(folder/"training.json", summary)
    model.eval()
    model.config.use_cache = True
    print("TRAINING_COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def environment(model):
    import platform
    import torch
    packages = {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "accelerate", "numpy", "Pillow", "matplotlib"]}
    return dict(python=sys.version, executable=sys.executable, prefix=sys.prefix,
                platform=platform.platform(), packages=packages, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), dtype=str(next(model.parameters()).dtype))


def rsi_loop(args, processor, model, qs, pool):
    import torch
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    memory, rounds = [], []
    for r in range(1, args.rounds+1):
        direct, _ = evaluate(args, processor, model, qs, f"rsi{r}_before")
        records, new_data, corrected = correct(args, processor, model, qs, direct,
                 f"rsi{r}_correction", self_check=True, round_idx=r)
        memory.extend(records)
        pool += new_data
        write_json(args.out/f"pool_round{r}.json", pool)
        write_json(args.out/"sft_data.json", pool)
        write_json(args.out/"rsi_memory.json", memory)
        trained = train_phase(args, processor, model, pool, f"rsi{r}_train", optimizer, len(new_data), r)
        _, end = evaluate(args, processor, model, qs, f"rsi{r}_after")
        row = dict(round=r, pool_size=len(pool), direct_correct=corrected["direct_correct"],
                   self_attempts=corrected["self_attempts"], self_rewrites=corrected["self_rewrites"],
                   tool_calls=corrected["tool_calls"], corrected_correct=corrected["correct"],
                   wrong_to_right=corrected["wrong_to_right"], right_to_wrong=corrected["right_to_wrong"],
                   new_accepted=len(new_data), new_used=trained["new_unique_samples_used"],
                   end_correct=end["correct"], end_accuracy=end["accuracy"], end_loc=end["loc_correct"],
                   end_loc_mean_iou=end["loc_mean_iou"])
        rounds.append(row)
        write_csv(args.out/"round_metrics.csv", rounds)
    return rounds


def summarize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stages = ["baseline", "sft_eval", "sft_harness", "rsi1_after", "rsi2_after"]
    summaries = [read_json(args.out/s/"metrics.json") for s in stages]
    fields = ["stage", "grid", "loc_correct", "loc_n", "count_correct", "count_n",
              "exist_correct", "exist_n", "correct", "n", "accuracy", "loc_mean_iou", "loc_parse_success"]
    write_csv(args.out/"metrics.csv", [{k: row[k] for k in fields} for row in summaries])
    rounds = list(csv.DictReader((args.out/"round_metrics.csv").open(encoding="utf-8-sig")))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    labels = ["Base", "SFT", "Harness", "RSI1", "RSI2"]
    axes[0].plot(labels, [s["loc_correct"]/8 for s in summaries], marker="o")
    axes[0].set(title=f"Grid {args.grid}: localization", ylabel="Accuracy", ylim=(-.03, 1.03))
    axes[1].plot(labels, [s["loc_mean_iou"] for s in summaries], marker="o")
    axes[1].set(title="Mean localization IoU", ylim=(-.03, 1.03))
    axes[2].plot([0, 1, 2], [15]+[int(r["pool_size"]) for r in rounds], marker="o")
    axes[2].set(title="Verified data pool", xlabel="RSI round", ylabel="Records", xticks=[0, 1, 2])
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(args.out/"rsi_curves.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    for phase in ["sft_train", "rsi1_train", "rsi2_train"]:
        rows = list(csv.DictReader((args.out/phase/"history.csv").open(encoding="utf-8-sig")))
        ax.plot([int(r["step"]) for r in rows], [float(r["loss"]) for r in rows], marker=".", label=phase)
    ax.set(xlabel="Optimizer step within phase", ylabel="Answer-token CE loss", title=f"Grid {args.grid}: training")
    ax.legend()
    ax.grid(alpha=.25)
    fig.savefig(args.out/"training_loss.png", dpi=180)
    plt.close(fig)


def full_run(args):
    import torch
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    gt = read_json(args.root/"hw6_imgs"/"gt.json")
    qs = build_questions(gt, args.grid)
    pool = seed_data(qs)
    write_json(args.out/"questions.json", qs)
    write_json(args.out/"pool_round0.json", pool)
    write_json(args.out/"sft_data.json", pool)
    config = read_json(Path(__file__).with_name("protocol.json"))
    config.update(grid=args.grid, actual_seed_steps=args.steps, actual_rounds=args.rounds,
                  model_path=str(args.model), data_manifest_sha256=sha_file(args.root/"hw6_imgs"/"manifest.json"),
                  model_weights_sha256=sha_file(Path(args.model)/"model.safetensors"),
                  source_sha256=sha_file(__file__))
    write_json(args.out/"config.json", config)
    processor, model = load_model(args)
    write_json(args.out/"environment.json", environment(model))
    evaluate(args, processor, model, qs, "baseline")
    model = attach_lora(model)
    trainable, total = model.get_nb_trainable_parameters()
    write_json(args.out/"parameters.json", dict(trainable=trainable, total=total,
                percent=100*trainable/total, initial_adapter_sha256=adapter_sha(model)))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    train_phase(args, processor, model, pool, "sft_train", optimizer)
    direct, initial = evaluate(args, processor, model, qs, "sft_eval")
    correct(args, processor, model, qs, direct, "sft_harness")
    rsi_loop(args, processor, model, qs, pool)
    summarize(args)
    write_json(args.out/"completion.json", dict(status="complete", seconds=time.perf_counter()-started,
                                               optimizer_steps=args.steps*(args.rounds+1)))
    print("RUN_COMPLETE", args.out, flush=True)


def check(args):
    from PIL import Image
    for grid in [4, 8]:
        for r in range(grid):
            for c in range(grid):
                assert parse_cell(cell_text(r, c), grid) == (r, c)
                assert parse_cell(f"r{r+1}c{c+1}", grid) == (r, c)
                assert parse_cell(f"{r+1},{c+1}", grid) == (r, c)
        assert parse_cell("第0行第1列", grid) is None
        assert parse_cell("我不知道", grid) is None
        assert parse_cell(f"第{grid+1}行第1列", grid) is None
        assert cell_of([0, 0, 10, 10], grid) == (0, 0)
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    gt = read_json(args.root/"hw6_imgs"/"gt.json")
    qs = build_questions(gt, args.grid)
    assert len(qs) == 15
    assert [sum(q["type"] == tp for q in qs) for tp in ["loc", "count", "exist"]] == [8, 4, 3]
    manifest = read_json(args.root/"hw6_imgs"/"manifest.json")
    for name, expected in manifest.items():
        assert sha_file(args.root/"hw6_imgs"/name) == expected
    # 验证器颜色相同但形状不同会误接收：保留工具边界，不改主评分。
    from PIL import ImageDraw
    im = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(im).rectangle([10, 10, 40, 40], outline=COLORS["红"], width=4)
    q = dict(target="红圆", cell=[3, 3], type="loc")
    passed, _, _ = verify_cell(im, q, (0, 0), 4)
    assert passed and not judge(q, "第1行第1列", 4)
    print("IMPLEMENTATION_CHECKS_PASSED parsers grids seeds manifests verifier_boundary", flush=True)


def model_check(args):
    """真实 GPU 前向/反向检查，不调用 optimizer.step，不生成实验权重。"""
    import torch
    proc, model = load_model(args)
    model = attach_lora(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    q = build_questions(read_json(args.root/"hw6_imgs"/"gt.json"), 4)[0]
    batch, plen = encode(proc, image(args.root, q), q["q"], q["gold"])
    batch = batch.to(next(model.parameters()).device)
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
    trainable, total = model.get_nb_trainable_parameters()
    result = dict(environment=environment(model), prompt_len=plen,
                  supervised_tokens=int(batch["labels"].ne(-100).sum()), loss=float(loss.detach()),
                  trainable=trainable, total=total, percent=100*trainable/total,
                  optimizer_updates=0, base_frozen=True, adapter_gradients_nonzero=True)
    write_json(args.root/"outputs"/"implementation_check.json", result)
    print("MODEL_CHECK_PASSED", json.dumps(result, ensure_ascii=False), flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["make_data", "make_sft", "make_seed", "check", "model_check", "run", "infer", "sft", "eval", "rsi", "summarize"], required=True)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("LAB06_ROOT", Path(__file__).resolve().parent.parent)))
    ap.add_argument("--model", default=os.environ.get("LAB06_MODEL", str(Path.home()/"llm_course_lab05/models/SmolVLM-256M-Instruct")))
    ap.add_argument("--grid", type=int, choices=[4, 8], default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    args.out = args.root/"outputs"/f"grid{args.grid}"
    if args.mode == "make_data":
        make_dataset(args.root)
        return
    if args.mode == "check":
        check(args)
        return
    if args.mode == "model_check":
        model_check(args)
        return
    if args.mode == "run":
        full_run(args)
        return
    if args.mode == "summarize":
        summarize(args)
        return
    qs = build_questions(read_json(args.root/"hw6_imgs"/"gt.json"), args.grid)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode in ["make_sft", "make_seed"]:
        assert not (args.out/"sft_data.json").exists()
        write_json(args.out/"sft_data.json", seed_data(qs))
        return
    if args.mode == "infer":
        proc, model = load_model(args)
        evaluate(args, proc, model, qs, "baseline")
    elif args.mode == "sft":
        import torch
        proc, model = load_model(args)
        model = attach_lora(model)
        pool = seed_data(qs)
        write_json(args.out/"pool_round0.json", pool)
        train_phase(args, proc, model, pool, "sft_train", torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4))
    else:
        proc, model = load_model(args, args.out/"sft_train"/"adapter")
        if args.mode == "eval":
            direct, _ = evaluate(args, proc, model, qs, "sft_eval")
            correct(args, proc, model, qs, direct, "sft_harness")
        else:
            rsi_loop(args, proc, model, qs, seed_data(qs))


if __name__ == "__main__":
    main()
