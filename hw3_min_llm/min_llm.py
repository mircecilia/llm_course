# -*- coding: utf-8 -*-
"""实验三：字符级最小 GPT，覆盖必做对照、采样分析与 3-gram 重复惩罚选做。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# 当前课程环境中 sklearn 先加载可避免 torch 与 matplotlib 的 OpenMP 运行库冲突。
import sklearn  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "lab03_outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SEED = 42


POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
    "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。",
    "远芳侵古道，晴翠接荒城。又送王孙去，萋萋满别情。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "鹅，鹅，鹅，曲项向天歌。白毛浮绿水，红掌拨清波。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。",
    "李白乘舟将欲行，忽闻岸上踏歌声。桃花潭水深千尺，不及汪伦送我情。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "天门中断楚江开，碧水东流至此回。两岸青山相对出，孤帆一片日边来。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。",
    "折戟沉沙铁未销，自将磨洗认前朝。东风不与周郎便，铜雀春深锁二乔。",
    "巴山楚水凄凉地，二十三年弃置身。怀旧空吟闻笛赋，到乡翻似烂柯人。",
    "沉舟侧畔千帆过，病树前头万木春。今日听君歌一曲，暂凭杯酒长精神。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
    "湖光秋月两相和，潭面无风镜未磨。遥望洞庭山水翠，白银盘里一青螺。",
    "杨柳青青江水平，闻郎江上踏歌声。东边日出西边雨，道是无晴却有晴。",
    "自古逢秋悲寂寥，我言秋日胜春朝。晴空一鹤排云上，便引诗情到碧霄。",
    "前不见古人，后不见来者。念天地之悠悠，独怆然而涕下。",
    "葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。",
    "黄河远上白云间，一片孤城万仞山。羌笛何须怨杨柳，春风不度玉门关。",
    "秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。",
    "月黑雁飞高，单于夜遁逃。欲将轻骑逐，大雪满弓刀。",
    "好雨知时节，当春乃发生。随风潜入夜，润物细无声。",
    "晓看红湿处，花重锦官城。野径云俱黑，江船火独明。",
    "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。",
    "烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。",
    "细草微风岸，危樯独夜舟。星垂平野阔，月涌大江流。",
    "风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。",
    "花近高楼伤客心，万方多难此登临。锦江春色来天地，玉垒浮云变古今。",
    "独怜幽草涧边生，上有黄鹂深树鸣。春潮带雨晚来急，野渡无人舟自横。",
    "天街小雨润如酥，草色遥看近却无。最是一年春好处，绝胜烟柳满皇都。",
    "昔人已乘黄鹤去，此地空余黄鹤楼。黄鹤一去不复返，白云千载空悠悠。",
    "晴川历历汉阳树，芳草萋萋鹦鹉洲。日暮乡关何处是，烟波江上使人愁。",
    "客舍青青柳色新，渭城朝雨浥轻尘。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "山光忽西落，池月渐东上。散发乘夕凉，开轩卧闲敞。",
    "荷笠带斜阳，青山独归远。苍苍竹林寺，杳杳钟声晚。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "人闲桂花落，夜静春山空。月出惊山鸟，时鸣春涧中。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "君自故乡来，应知故乡事。来日绮窗前，寒梅著花未。",
    "山中相送罢，日暮掩柴扉。春草明年绿，王孙归不归。",
    "花间一壶酒，独酌无相亲。举杯邀明月，对影成三人。",
    "小时不识月，呼作白玉盘。又疑瑶台镜，飞在青云端。",
    "长安一片月，万户捣衣声。秋风吹不尽，总是玉关情。",
    "弃我去者，昨日之日不可留。乱我心者，今日之日多烦忧。",
    "抽刀断水水更流，举杯消愁愁更愁。人生在世不称意，明朝散发弄扁舟。",
    "千里黄云白日曛，北风吹雁雪纷纷。莫愁前路无知己，天下谁人不识君。",
    "慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。谁言寸草心，报得三春晖。",
    "山重水复疑无路，柳暗花明又一村。莫笑农家腊酒浑，丰年留客足鸡豚。",
    "纸上得来终觉浅，绝知此事要躬行。古人学问无遗力，少壮功夫老始成。",
    "死去元知万事空，但悲不见九州同。王师北定中原日，家祭无忘告乃翁。",
    "小荷才露尖尖角，早有蜻蜓立上头。泉眼无声惜细流，树阴照水爱晴柔。",
    "接天莲叶无穷碧，映日荷花别样红。毕竟西湖六月中，风光不与四时同。",
    "胜日寻芳泗水滨，无边光景一时新。等闲识得东风面，万紫千红总是春。",
    "半亩方塘一鉴开，天光云影共徘徊。问渠那得清如许，为有源头活水来。",
    "郁孤台下清江水，中间多少行人泪。西北望长安，可怜无数山。",
    "人生自古谁无死，留取丹心照汗青。辛苦遭逢起一经，干戈寥落四周星。",
    "咬定青山不放松，立根原在破岩中。千磨万击还坚劲，任尔东西南北风。",
    "千锤万凿出深山，烈火焚烧若等闲。粉骨碎身浑不怕，要留清白在人间。",
    "浩荡离愁白日斜，吟鞭东指即天涯。落红不是无情物，化作春泥更护花。",
    "九州生气恃风雷，万马齐喑究可哀。我劝天公重抖擞，不拘一格降人才。",
    "力微任重久神疲，再竭衰庸定不支。苟利国家生死以，岂因祸福避趋之。",
]


def build_corpus(extra_file: Path | None = None) -> str:
    text = "\n".join(POEMS)
    if extra_file is not None and extra_file.exists():
        text += "\n" + extra_file.read_text(encoding="utf-8")
    return text


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(chars)}
        self.itos = {index: char for char, index in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text if char in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[index] for index in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels = inputs.shape
        query, key, value = self.qkv(inputs).split(channels, dim=2)
        query = query.view(batch_size, time_steps, self.n_head, -1).transpose(1, 2)
        key = key.view(batch_size, time_steps, self.n_head, -1).transpose(1, 2)
        value = value.view(batch_size, time_steps, self.n_head, -1).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(key.size(-1))
        attention = attention.masked_fill(
            self.mask[:, :, :time_steps, :time_steps] == 0, float("-inf")
        )
        attention = F.softmax(attention, dim=-1)
        outputs = attention @ value
        outputs = outputs.transpose(1, 2).contiguous().view(
            batch_size, time_steps, channels
        )
        return self.proj(outputs)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.ln1(inputs))
        return inputs + self.mlp(self.ln2(inputs))


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 2,
        block_size: int = 128,
        use_pos: bool = True,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = (
            nn.Embedding(block_size, n_embd) if use_pos else None
        )
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, indices: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time_steps = indices.shape
        positions = torch.arange(time_steps, device=indices.device)
        hidden = self.token_embedding(indices)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        indices: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int | None = 20,
        no_repeat_ngram_size: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            conditioned = indices[:, -self.block_size :]
            logits, _ = self(conditioned)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if no_repeat_ngram_size:
                _ban_repeated_ngram_tokens(logits, indices, no_repeat_ngram_size)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, 1)
            indices = torch.cat([indices, next_id], dim=1)
        return indices


def _ban_repeated_ngram_tokens(
    logits: torch.Tensor, indices: torch.Tensor, ngram_size: int
) -> None:
    if indices.size(1) < ngram_size - 1:
        return
    for batch_index in range(indices.size(0)):
        tokens = indices[batch_index].tolist()
        prefix = tuple(tokens[-(ngram_size - 1) :])
        banned_by_prefix: dict[tuple[int, ...], set[int]] = {}
        for index in range(len(tokens) - ngram_size + 1):
            old_prefix = tuple(tokens[index : index + ngram_size - 1])
            next_token = tokens[index + ngram_size - 1]
            banned_by_prefix.setdefault(old_prefix, set()).add(next_token)
        if prefix in banned_by_prefix:
            logits[batch_index, list(banned_by_prefix[prefix])] = float("-inf")


@dataclass(frozen=True)
class ExperimentConfig:
    key: str
    group: str
    change: str
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    block_size: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 32
    iterations: int = 1000
    use_pos: bool = True


BASELINE = ExperimentConfig(
    key="baseline",
    group="baseline",
    change="基线：L=2, C=128, H=4, T=128, lr=1e-3, 有位置编码",
    iterations=2000,
)

EXPERIMENTS = [
    BASELINE,
    replace(BASELINE, key="exp1_lr1e2", group="exp1", change="lr: 1e-3 -> 1e-2", learning_rate=1e-2, iterations=1000),
    replace(BASELINE, key="exp1_lr1e4", group="exp1", change="lr: 1e-3 -> 1e-4", learning_rate=1e-4, iterations=1000),
    replace(BASELINE, key="exp2_layer1", group="exp2", change="n_layer: 2 -> 1", n_layer=1, iterations=1000),
    replace(BASELINE, key="exp2_layer4", group="exp2", change="n_layer: 2 -> 4", n_layer=4, iterations=1000),
    replace(BASELINE, key="exp3_embd64", group="exp3", change="n_embd: 128 -> 64; n_head: 4 -> 2", n_embd=64, n_head=2, iterations=1000),
    replace(BASELINE, key="exp3_embd256", group="exp3", change="n_embd: 128 -> 256; n_head: 4 -> 8", n_embd=256, n_head=8, iterations=1000),
    replace(BASELINE, key="exp4_block32", group="exp4", change="block_size: 128 -> 32", block_size=32, iterations=1000),
    replace(BASELINE, key="exp5_no_pos", group="exp5", change="移除可学习位置编码", use_pos=False, iterations=1000),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        len(data) - block_size - 1, (batch_size,), generator=generator
    )
    inputs = torch.stack([data[index : index + block_size] for index in starts])
    targets = torch.stack(
        [data[index + 1 : index + block_size + 1] for index in starts]
    )
    return inputs, targets


def build_model(config: ExperimentConfig, vocab_size: int) -> MiniGPT:
    return MiniGPT(
        vocab_size=vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        use_pos=config.use_pos,
    )


def parameter_estimate(config: ExperimentConfig, vocab_size: int) -> int:
    position = config.block_size * config.n_embd if config.use_pos else 0
    return (
        vocab_size * config.n_embd
        + position
        + config.n_layer * 12 * config.n_embd**2
    )


def curve_name(config: ExperimentConfig) -> str:
    if config.key == "baseline":
        return "loss_curve_L2_E128_lr0.001.png"
    return f"loss_curve_{config.key}.png"


CURVE_TITLES = {
    "baseline": "Baseline: L=2, C=128, H=4, T=128, lr=1e-3, position=on",
    "exp1_lr1e2": "Learning rate: 1e-3 -> 1e-2",
    "exp1_lr1e4": "Learning rate: 1e-3 -> 1e-4",
    "exp2_layer1": "Transformer layers: 2 -> 1",
    "exp2_layer4": "Transformer layers: 2 -> 4",
    "exp3_embd64": "Embedding: 128 -> 64; heads: 4 -> 2",
    "exp3_embd256": "Embedding: 128 -> 256; heads: 4 -> 8",
    "exp4_block32": "Context length: 128 -> 32",
    "exp5_no_pos": "Position embedding: on -> off",
}


def save_curve(config: ExperimentConfig, losses: list[float]) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(1, len(losses) + 1), losses, linewidth=0.8)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Cross-Entropy Loss")
    axis.set_title(CURVE_TITLES[config.key])
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / curve_name(config), dpi=160, bbox_inches="tight")
    plt.close(figure)


def text_metrics(text: str) -> dict[str, float]:
    content = text[1:] if text else text
    trigrams = [content[index : index + 3] for index in range(max(0, len(content) - 2))]
    repeated_rate = 0.0
    if trigrams:
        repeated_rate = 1.0 - len(set(trigrams)) / len(trigrams)
    return {
        "unique_char_ratio": len(set(content)) / max(1, len(content)),
        "repeated_trigram_rate": repeated_rate,
    }


def generate_samples(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompts: list[str],
    samples_each: int,
    temperature: float,
    top_k: int | None,
    max_new_tokens: int = 200,
    no_repeat_ngram_size: int | None = None,
    seed_start: int = 1000,
) -> list[dict[str, object]]:
    rows = []
    sample_number = 0
    for prompt in prompts:
        for _ in range(samples_each):
            sample_number += 1
            sample_seed = seed_start + sample_number
            torch.manual_seed(sample_seed)
            encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            generated = model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            text = tokenizer.decode(generated[0].tolist())
            rows.append(
                {
                    "prompt": prompt,
                    "seed": sample_seed,
                    "text": text,
                    **text_metrics(text),
                }
            )
    return rows


def save_generated_text(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    sections = [title]
    for index, row in enumerate(rows, start=1):
        sections.extend(
            [
                "",
                f"--- 样本 {index} | prompt={row['prompt']} | seed={row['seed']} ---",
                str(row["text"]),
            ]
        )
    path.write_text("\n".join(sections), encoding="utf-8")


def train_experiment(
    config: ExperimentConfig,
    tokenizer: CharTokenizer,
    data: torch.Tensor,
    force: bool = False,
) -> dict[str, object]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    metrics_path = OUTPUT_DIR / f"metrics_{config.key}.json"
    if metrics_path.exists() and not force:
        print(f"[跳过] {config.key} 已有完整结果")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    set_seed(SEED)
    model = build_model(config, tokenizer.vocab_size)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_generator = torch.Generator().manual_seed(SEED)
    initial_generator = torch.Generator().manual_seed(SEED + 999)
    initial_inputs, initial_targets = get_batch(
        data, config.block_size, config.batch_size, initial_generator
    )
    model.eval()
    with torch.no_grad():
        _, initial_loss_tensor = model(initial_inputs, initial_targets)
    initial_loss = float(initial_loss_tensor.item())

    print(f"\n===== {config.key}: {config.change} =====")
    print(
        f"参数量 {parameter_count:,} | 初始 loss {initial_loss:.4f} | "
        f"ln(V)={math.log(tokenizer.vocab_size):.4f}"
    )
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, config.iterations + 1):
        inputs, targets = get_batch(
            data, config.block_size, config.batch_size, train_generator
        )
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if step == 1 or step % 100 == 0:
            elapsed = time.perf_counter() - started
            speed = step / elapsed
            remaining = (config.iterations - step) / speed
            print(
                f"step {step:4d}/{config.iterations} | loss {loss.item():.4f} | "
                f"{speed:.2f} it/s | 剩余约 {remaining:.0f}s",
                flush=True,
            )
    training_seconds = time.perf_counter() - started

    save_curve(config, losses)
    generation_rows = generate_samples(
        model,
        tokenizer,
        prompts=["春"],
        samples_each=2,
        temperature=1.0,
        top_k=20,
        max_new_tokens=120,
        seed_start=2000,
    )
    save_generated_text(
        OUTPUT_DIR / f"generated_{config.key}.txt",
        f"{config.key} | {config.change}",
        generation_rows,
    )
    if config.key == "baseline":
        baseline_rows = generate_samples(
            model,
            tokenizer,
            prompts=["春", "月"],
            samples_each=2,
            temperature=1.0,
            top_k=20,
            max_new_tokens=200,
            seed_start=3000,
        )
        save_generated_text(
            OUTPUT_DIR / "generated_base.txt",
            "基线生成：temperature=1.0, top_k=20",
            baseline_rows,
        )
        torch.save(model.state_dict(), CHECKPOINT_DIR / "baseline_state.pt")

    metrics = {
        **asdict(config),
        "vocab_size": tokenizer.vocab_size,
        "corpus_characters": len(data),
        "parameters": parameter_count,
        "parameter_estimate": parameter_estimate(config, tokenizer.vocab_size),
        "initial_loss": initial_loss,
        "ln_vocab": math.log(tokenizer.vocab_size),
        "final_loss": losses[-1],
        "last100_mean_loss": sum(losses[-100:]) / min(100, len(losses)),
        "training_seconds": training_seconds,
        "curve_file": curve_name(config),
        "generation_file": f"generated_{config.key}.txt",
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / f"history_{config.key}.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "loss"])
        writer.writerows(enumerate(losses, start=1))
    print(
        f"完成 {config.key}: final loss={losses[-1]:.4f}, "
        f"last100={metrics['last100_mean_loss']:.4f}, "
        f"耗时={training_seconds / 60:.1f} 分钟"
    )
    return metrics


def load_baseline(
    tokenizer: CharTokenizer,
) -> MiniGPT:
    model = build_model(BASELINE, tokenizer.vocab_size)
    state = torch.load(
        CHECKPOINT_DIR / "baseline_state.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model


def run_sampling_and_optional(tokenizer: CharTokenizer) -> None:
    model = load_baseline(tokenizer)
    sampling_configs = [
        ("temp0.5_top20", 0.5, 20),
        ("temp1.0_top20", 1.0, 20),
        ("temp1.5_top20", 1.5, 20),
        ("temp1.0_top5", 1.0, 5),
        ("temp1.0_unlimited", 1.0, None),
    ]
    all_rows: list[dict[str, object]] = []
    text_sections = []
    for config_index, (name, temperature, top_k) in enumerate(sampling_configs):
        rows = generate_samples(
            model,
            tokenizer,
            prompts=["春", "月", "山"],
            samples_each=1,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=200,
            seed_start=4000 + config_index * 10,
        )
        for row in rows:
            row.update(
                {
                    "config": name,
                    "temperature": temperature,
                    "top_k": "不限" if top_k is None else top_k,
                }
            )
        all_rows.extend(rows)
        text_sections.append(f"===== {name} =====")
        for row in rows:
            text_sections.append(f"prompt={row['prompt']} seed={row['seed']}")
            text_sections.append(str(row["text"]))
            text_sections.append("")

    sampling_fields = [
        "config",
        "temperature",
        "top_k",
        "prompt",
        "seed",
        "unique_char_ratio",
        "repeated_trigram_rate",
        "text",
    ]
    with (OUTPUT_DIR / "sampling_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=sampling_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (OUTPUT_DIR / "generated_sampling.txt").write_text(
        "\n".join(text_sections), encoding="utf-8"
    )

    comparison_rows = []
    comparison_text = [
        "固定基线模型与随机种子；temperature=0.5, top_k=1, 生成400字。",
        "唯一变量：是否禁止重复 3-gram。",
        "",
    ]
    for prompt_index, prompt in enumerate(["春", "月", "山"]):
        shared_seed = 5001 + prompt_index
        for label, ngram_size in [("原始采样", None), ("禁止重复3-gram", 3)]:
            row = generate_samples(
                model,
                tokenizer,
                prompts=[prompt],
                samples_each=1,
                temperature=0.5,
                top_k=1,
                max_new_tokens=400,
                no_repeat_ngram_size=ngram_size,
                seed_start=shared_seed - 1,
            )[0]
            row.update(
                {
                    "method": label,
                    "no_repeat_ngram_size": ngram_size or 0,
                    "temperature": 0.5,
                    "top_k": 1,
                    "max_new_tokens": 400,
                }
            )
            comparison_rows.append(row)
            comparison_text.extend(
                [f"===== {prompt} | {label} | seed={row['seed']} =====", str(row["text"]), ""]
            )
    with (OUTPUT_DIR / "repetition_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fields = [
            "method",
            "no_repeat_ngram_size",
            "temperature",
            "top_k",
            "max_new_tokens",
            "prompt",
            "seed",
            "unique_char_ratio",
            "repeated_trigram_rate",
            "text",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    (OUTPUT_DIR / "generated_repetition_compare.txt").write_text(
        "\n".join(comparison_text), encoding="utf-8"
    )


def write_results(metrics_rows: list[dict[str, object]]) -> None:
    fields = [
        "group",
        "key",
        "change",
        "n_layer",
        "n_embd",
        "n_head",
        "block_size",
        "learning_rate",
        "use_pos",
        "iterations",
        "vocab_size",
        "parameters",
        "initial_loss",
        "final_loss",
        "last100_mean_loss",
        "training_seconds",
        "curve_file",
        "generation_file",
    ]
    with (OUTPUT_DIR / "results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics_rows)


def run_full_suite(force: bool = False) -> list[dict[str, object]]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    text = build_corpus()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(
        f"PyTorch {torch.__version__} | CPU | 线程数 {torch.get_num_threads()} | "
        f"语料 {len(text)} 字符 | V={tokenizer.vocab_size}"
    )
    metrics_rows = [
        train_experiment(config, tokenizer, data, force=force)
        for config in EXPERIMENTS
    ]
    write_results(metrics_rows)
    run_sampling_and_optional(tokenizer)

    baseline = metrics_rows[0]
    parameter_check = {
        "formula": "V*C + T*C + L*12*C^2",
        "substitution": "699*128 + 128*128 + 2*12*128^2",
        "estimate": parameter_estimate(BASELINE, tokenizer.vocab_size),
        "exact": baseline["parameters"],
        "relative_error_percent": abs(
            parameter_estimate(BASELINE, tokenizer.vocab_size)
            - int(baseline["parameters"])
        )
        / int(baseline["parameters"])
        * 100,
    }
    (OUTPUT_DIR / "parameter_check.json").write_text(
        json.dumps(parameter_check, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n全部必做训练、采样对照与重复惩罚选做已完成。")
    return metrics_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验三：手搓最小 LLM")
    parser.add_argument("--suite", action="store_true", help="运行全部必做和选做")
    parser.add_argument("--force", action="store_true", help="覆盖已有实验三结果")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_pos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.suite:
        run_full_suite(force=args.force)
        return
    text = build_corpus()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    config = replace(
        BASELINE,
        iterations=args.iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        learning_rate=args.lr,
        use_pos=not args.no_pos,
    )
    metrics = train_experiment(config, tokenizer, data, force=args.force)
    write_results([metrics])


if __name__ == "__main__":
    main()
