# -*- coding: utf-8 -*-
"""CPU distillation based on the course guide, sections 7.1-7.6."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import platform
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
ROOT = Path(__file__).resolve().parent
# ============================================================
# 1. 语料：可复现的合成「仿古诗句」
#  四个槽位  + 两个收尾模板，使下一个字的分布具有真实的多样性与结构，
#  这正是教师  soft label 能传给学生的「暗知识」。
# ============================================================
SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人",
        "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡",
      "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看",
        "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟",
        "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声",
        "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]
POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
]
def build_corpus(n_lines=3000, seed=0):
    """用固定随机种子生成训练语料；留出集换一个种子，检验泛化而非记忆。"""
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append("%s%s%s%s，%s。" % (s, p, v, o, m))
        elif i % 3 == 1:
            out.append("%s在%s%s了一夜%s，%s。" % (s, p, o, rng.choice(["看", "听", "守", "忆"]),
m))
        else:
            out.append("%s%s道：“%s%s。”%s。" % (s, v, o, rng.choice(["正好", "可怜", "可惜", "依旧"]), m))
    out.append("\n")
    out.extend(POEMS)
    text = "".join(out)
    return text
# ============================================================
# 2. 字符级分词
# ============================================================
class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)
    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]
    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)
# ============================================================
# 3. 最小 GPT（Pre-Norm，与作业三同构，只改宽度/深度）
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        assert C % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.drop = nn.Dropout(cfg["dropout"])
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"]))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)
        self.drop = nn.Dropout(cfg["dropout"])
    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))
class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.apply(self._init)
        self.n_param = sum(p.numel() for p in self.parameters())
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None, need_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        if need_hidden:
            return logits, loss, x
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
# ============================================================
# 4. 批采样：scratch 与 kd 使用同一 generator 种子，保证数据顺序一致
# ============================================================
def get_batch(data, bs, block, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y
@torch.no_grad()
def evaluate(model, data, bs, block, iters=20, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, block, gen)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters
@torch.no_grad()
def agreement(student, teacher, data, bs, block, iters=10, seed=4321):
    """学生与教师在留出集上的 top-1 一致率，衡量「学到了多少」"""
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    hit = tot = 0
    for _ in range(iters):
        x, _ = get_batch(data, bs, block, gen)
        a = student(x)[0].argmax(-1)
        b = teacher(x)[0].argmax(-1)
        hit += (a == b).sum().item()
        tot += b.numel()
    student.train()
    teacher.train()
    return hit / tot
# ============================================================
# 5. 训练：hard label / soft label（KD）
# ============================================================
def train(model, data, iters, bs, block, lr=3e-3, seed=0,
       teacher=None, T=4.0, alpha=0.5, feat_weight=0.0,
       log_every=50, tag=""):
    """
    alpha = 0 -> 纯交叉熵（从零训练的对照组）
    alpha > 0 -> L = (1-alpha)*CE + alpha*T^2*KL(教师软标签 || 学生)
    feat_weight -> 额外加最后一层隐状态 MSE（FitNet 式特征蒸馏）
    """
    proj = None
    if teacher is not None and feat_weight > 0:
        proj = nn.Linear(model.cfg["n_embd"], teacher.cfg["n_embd"])
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    hist = []
    if teacher is not None:
        teacher.eval()
    model.train()
    t0 = time.time()
    for it in range(iters):
        x, y = get_batch(data, bs, block, gen)
        logits = model(x)[0]
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        loss = ce
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                t_logits = teacher(x)[0]
            # 关键：先 reshape 成 (B*T, V) 再算 KL。
            # 若直接对 (B,T,V) 调用 kl_div(reduction="batchmean")，它只除以 B，
            # 序列维会被重复累加，KL 被放大 T 倍（T=96 时数值大得离谱）。
            V = logits.size(-1)
            s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
            t_p = F.softmax(t_logits.reshape(-1, V) / T, dim=-1)
            kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
            loss = (1 - alpha) * ce + alpha * kd
        if proj is not None:
            _, _, hs = model(x, need_hidden=True)
            with torch.no_grad():
                _, _, ht = teacher(x, need_hidden=True)
            loss = loss + feat_weight * F.mse_loss(proj(hs), ht)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        if (it + 1) % log_every == 0 or it == 0:
            extra = ""
            if teacher is not None and alpha > 0:
                extra = " ce %.4f kd(T^2) %.4f" % (ce.item(), kd.item())
            print(" [%s] iter %4d/%d loss %.4f%s (%.1fs)"
                    % (tag or "train", it + 1, iters, loss.item(), extra, time.time() - t0))
    return hist, time.time() - t0
# ============================================================
# 6. 主流程
# ============================================================
TEACHER_CFG = dict(n_layer=2, n_head=4, n_embd=160, block_size=96, dropout=0.0)
STUDENT_CFG = dict(n_layer=2, n_head=4, n_embd=64, block_size=96, dropout=0.0)
def make_data(block):
    train_text = build_corpus(3000, seed=0)
    val_text = build_corpus(300, seed=999) # 同语法、不同种子 -> 留出集
    tok = Tokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return tok, train_ids, val_ids


def new_model(tokenizer, config, seed):
    torch.manual_seed(seed)
    return GPT(tokenizer.vocab_size, config)


def metric(model, val, args, curve, seconds, teacher=None):
    loss = evaluate(model, val, args.batch_size, TEACHER_CFG["block_size"])
    row = dict(params=model.n_param, steps=len(curve), val_loss=loss,
               ppl=math.exp(loss), time_s=seconds, curve=curve,
               last_train_objective=curve[-1])
    if teacher is not None:
        row["agree"] = agreement(model, teacher, val, args.batch_size, TEACHER_CFG["block_size"])
    return row


def show(label, row):
    print(f"[{label}] params={row['params']} val_loss={row['val_loss']:.6f} "
          f"ppl={row['ppl']:.4f} train={row['time_s']:.2f}s", flush=True)


def sample(model, tokenizer):
    torch.manual_seed(31415)
    prompt = torch.tensor([tokenizer.encode("月")], dtype=torch.long)
    tokens = model.generate(prompt, 60, temperature=0.9, top_k=20)[0].tolist()
    return tokenizer.decode(tokens[1:])


def train_teacher(tokenizer, data, val, args, width):
    teacher = new_model(tokenizer, dict(TEACHER_CFG, n_embd=width), args.seed)
    curve, elapsed = train(teacher, data, args.teacher_iters, args.batch_size,
                           TEACHER_CFG["block_size"], lr=args.lr, seed=args.seed,
                           log_every=args.log_every, tag=f"teacher-{width}")
    row = metric(teacher, val, args, curve, elapsed)
    show(f"teacher-{width}", row)
    return teacher, row


def train_student(label, tokenizer, data, val, args, reference, kd=False, feat=0.0):
    student = new_model(tokenizer, STUDENT_CFG, args.seed + 7)
    # 两组使用同一初始化与同一批次 RNG。哈希与采样配置随结果保存。
    initial = hashlib.sha256(b"".join(p.detach().numpy().tobytes()
                                     for p in student.parameters())).hexdigest()
    curve, elapsed = train(student, data, args.student_iters, args.batch_size,
                           TEACHER_CFG["block_size"], lr=args.lr, seed=args.seed,
                           teacher=reference if kd else None, T=args.T,
                           alpha=args.alpha if kd else 0.0, feat_weight=feat,
                           log_every=args.log_every, tag=label)
    row = metric(student, val, args, curve, elapsed, reference)
    row.update(initial_sha256=initial, batch_seed=args.seed + 1)
    show(label, row)
    return student, row


def sweep(teacher, tokenizer, data, val, args):
    gen = torch.Generator().manual_seed(2024)
    block = TEACHER_CFG["block_size"]
    teacher.eval()
    cache = []
    with torch.no_grad():
        for _ in range(32):
            x, y = get_batch(data, args.batch_size, block, gen)
            cache.append((x, y, teacher(x)[0]))
    rows = []
    for temperature in (1, 2, 4, 8):
        for alpha in (0.3, 0.5, 0.7, 0.9):
            model = new_model(tokenizer, STUDENT_CFG, args.seed + 7)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            targets = [F.softmax(logits.flatten(0, 1) / temperature, dim=-1)
                       for _, _, logits in cache]
            started = time.perf_counter()
            for step in range(args.abl_iters):
                i = step % len(cache)
                x, y, _ = cache[i]
                logits = model(x)[0].flatten(0, 1)
                ce = F.cross_entropy(logits, y.reshape(-1))
                kl = F.kl_div(F.log_softmax(logits / temperature, dim=-1),
                              targets[i], reduction="batchmean")
                loss = (1 - alpha) * ce + alpha * temperature**2 * kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            elapsed = time.perf_counter() - started
            value = evaluate(model, val, args.batch_size, block)
            row = dict(T=temperature, alpha=alpha, steps=args.abl_iters,
                       val_loss=value, ppl=math.exp(value), time_s=elapsed)
            rows.append(row)
            print(f"[ablate] T={temperature} alpha={alpha:.1f} val={value:.6f} "
                  f"train={elapsed:.2f}s", flush=True)
    return rows


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def plot(out, tag, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if tag == "all":
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for label, key in (("Teacher (CE)", "teacher"), ("Scratch (CE)", "scratch"),
                           ("KD (mixed objective)", "kd")):
            values = rows[key]["curve"]
            ax.plot(range(1, len(values) + 1), values, label=label, linewidth=1.2)
        ax.set(xlabel="Optimizer step", ylabel="Training objective")
        ax.grid(alpha=0.2)
        ax.legend()
        name = "curves.png"
    else:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        temperatures, alphas = (1, 2, 4, 8), (0.3, 0.5, 0.7, 0.9)
        lookup = {(r["T"], r["alpha"]): r["val_loss"] for r in rows}
        grid = [[lookup[t, a] for a in alphas] for t in temperatures]
        picture = ax.imshow(grid, cmap="viridis_r", aspect="auto")
        best = min(lookup, key=lookup.get)
        mid = (min(lookup.values()) + max(lookup.values())) / 2
        for i, t in enumerate(temperatures):
            for j, a in enumerate(alphas):
                v = lookup[t, a]
                ax.text(j, i, f"{v:.4f}" + (" *" if (t, a) == best else ""),
                        ha="center", va="center", color="white" if v > mid else "black")
        ax.set(xticks=range(4), xticklabels=alphas, yticks=range(4),
               yticklabels=temperatures, xlabel="Alpha", ylabel="Temperature")
        fig.colorbar(picture, ax=ax, label="Held-out CE")
        name = "heatmap.png"
    fig.tight_layout()
    fig.savefig(out / name, dpi=180)
    plt.close(fig)


def run(args):
    started = time.perf_counter()
    out = args.output
    mode = args.mode
    if mode == "quick":
        out = out / "quick"
        args.teacher_iters, args.student_iters, args.batch_size = 30, 20, 8
        mode = "all"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"results_{args.mode}.json"
    if target.exists() and not (mode == "teacher" and args.retrain):
        raise FileExistsError(f"Result already exists: {target}. Use --output for an independent run.")
    tokenizer, data, val = make_data(TEACHER_CFG["block_size"])
    meta = dict(vocab=tokenizer.vocab_size, train_chars=len(data), val_chars=len(val),
                python=platform.python_version(), torch=torch.__version__,
                torch_cuda=torch.version.cuda, device="cpu", threads=torch.get_num_threads(),
                seed=args.seed, lr=args.lr, batch=args.batch_size, teacher_steps=args.teacher_iters,
                student_steps=args.student_iters, ablation_steps=args.abl_iters,
                T=args.T, alpha=args.alpha, teacher_config=TEACHER_CFG, student_config=STUDENT_CFG,
                corpus_train_seed=0, corpus_val_seed=999, shared_fixed_poems=len(POEMS))
    print("[protocol]", json.dumps(meta, ensure_ascii=False), flush=True)
    checkpoint = out / "teacher.pt"
    if mode == "teacher" or args.mode == "quick":
        teacher, teacher_row = train_teacher(tokenizer, data, val, args, 160)
        torch.save(teacher.state_dict(), checkpoint)
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Train the teacher first: {checkpoint}")
        saved = json.loads((out / "results_teacher.json").read_text(encoding="utf-8"))
        for key in ("vocab", "seed", "lr", "batch", "teacher_steps", "teacher_config"):
            if saved["protocol"][key] != meta[key]:
                raise ValueError(f"Teacher protocol mismatch: {key}")
        teacher_row = saved["teacher"]
        teacher = new_model(tokenizer, TEACHER_CFG, args.seed)
        teacher.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    result = dict(protocol=meta, teacher=teacher_row)
    if mode == "all":
        scratch, result["scratch"] = train_student("scratch", tokenizer, data, val, args, teacher)
        kd, result["kd"] = train_student("kd", tokenizer, data, val, args, teacher, kd=True)
        assert result["scratch"]["initial_sha256"] == result["kd"]["initial_sha256"]
        if args.mode != "quick":
            result["samples"] = {k: sample(m, tokenizer) for k, m in
                                 (("teacher", teacher), ("scratch", scratch), ("kd", kd))}
            torch.save(scratch.state_dict(), out / "student_scratch.pt")
            torch.save(kd.state_dict(), out / "student_kd.pt")
            plot(out, "all", result)
    elif mode == "ablate":
        result["ablation"] = sweep(teacher, tokenizer, data, val, args)
        plot(out, "ablate", result["ablation"])
    elif mode == "scale":
        main = json.loads((out / "results_all.json").read_text(encoding="utf-8"))
        for key in ("seed", "lr", "batch", "teacher_steps", "student_steps", "T", "alpha"):
            if main["protocol"][key] != meta[key]:
                raise ValueError(f"Main comparison protocol mismatch: {key}")
        result["scratch_val_loss"] = main["scratch"]["val_loss"]
        result["scale"] = []
        for width in (96, 160, 256):
            if width == 160:
                trow, srow = teacher_row, main["kd"]
            else:
                ref, trow = train_teacher(tokenizer, data, val, args, width)
                torch.save(ref.state_dict(), out / f"teacher_{width}.pt")
                _, srow = train_student(f"kd-{width}", tokenizer, data, val, args, ref, kd=True)
            assert srow["initial_sha256"] == main["scratch"]["initial_sha256"]
            result["scale"].append(dict(width=width, teacher=trow, kd=srow))
            print(f"[scale] width={width} teacher={trow['val_loss']:.6f} "
                  f"kd={srow['val_loss']:.6f}", flush=True)
    result["phase_wall_s"] = time.perf_counter() - started
    write_json(target, result)
    print(f"[done] {target.name} {result['phase_wall_s']:.2f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=("quick", "teacher", "all", "ablate", "scale"))
    ap.add_argument("--output", type=Path, default=ROOT / "runs_distill")
    ap.add_argument("--teacher-iters", type=int, default=200)
    ap.add_argument("--student-iters", type=int, default=120)
    ap.add_argument("--abl-iters", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--T", type=float, default=4)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=40)
    ap.add_argument("--retrain", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
