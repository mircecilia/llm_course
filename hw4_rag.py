# -*- coding: utf-8 -*-
"""实验四：同一文档集上的 Vector RAG、GraphRAG、WikiRAG 检索对比。"""

import argparse
import csv
import json
import platform
import time
import textwrap
from collections import defaultdict, deque
from pathlib import Path

from hw4_rag_api import ANSWER_PROMPT, llm_chat

import sklearn  # 在本机环境中先加载 sklearn，避免 OpenMP 运行库冲突
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import networkx as nx
import numpy as np
import torch
import sentence_transformers
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lab04_outputs"
MODEL = "BAAI/bge-small-zh-v1.5"

# 指南给出的 15 篇虚构校园文档；所有检索器使用同一份原文。
CORPUS = {
    "doc01": "云山大学位于岭南省云州市，建于 1958 年，是省属重点大学，现有全日制本科生约 2.1 万人。校训为「格物致知」。",
    "doc02": "人工智能学院成立于 2018 年，首任及现任院长为王启明教授。学院下设机器学习系、智能科学系、认知计算系三个系。",
    "doc03": "计算机学院前身为 1985 年成立的计算机系，现任院长陈国峰教授，设计算机科学与技术、软件工程两个本科专业。",
    "doc04": "人工智能学院教师：李文瀚，教授，2019 年入职，研究方向为检索增强生成，主讲《大模型通识课》；赵婉晴，副教授，研究方向为知识图谱，主讲《知识图谱导论》；孙浩然，讲师，研究方向为强化学习。",
    "doc05": "计算机学院教师：周天宇，教授，研究方向为分布式系统；林小雨，副教授，研究方向为数据库系统。",
    "doc06": "《大模型通识课》课程代码 AI2101，3 学分、32 学时，春季学期开设，授课教师李文瀚，面向全校本科生，无先修课程要求。",
    "doc07": "《知识图谱导论》课程代码 AI3305，2 学分，秋季学期开设，授课教师赵婉晴，面向人工智能学院研究生。",
    "doc08": "选课规则：本科生每学期最多修 30 学分；GPA 低于 2.0 给予学术警告。先修要求：《知识图谱导论》需先修《数据结构》，《大模型通识课》无先修要求。",
    "doc09": "奖学金体系：国家奖学金 8000 元/年，评定比例约 2%；校长奖学金 20000 元/年，全校每年 10 人；云山一等奖学金 3000 元/年，比例约 5%。",
    "doc10": "科研平台：认知计算全国重点实验室依托人工智能学院建设；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理。",
    "doc11": "学生社团：AI 协会，指导教师李文瀚，每周三晚组织论文研读；机器人战队，指导教师孙浩然，每年参加全国机器人大赛。",
    "doc12": "校历：春季学期 3 月 2 日开学、7 月 5 日放暑假；秋季学期 9 月 1 日开学、次年 1 月 15 日放寒假。",
    "doc13": "图书馆藏书 380 万册，开放时间为每天 7:00-22:00；人工智能分馆位于理科楼 B 座 3 层，收藏大模型与智能体专题图书。",
    "doc14": "校园交通：地铁 3 号线「云大站」距东门 200 米；校内校巴共 5 条线路，10 分钟一班。",
    "doc15": "国际交流：学校与 12 个国家的 47 所高校签有交换协议；人工智能学院与新加坡南洋理工大学有本科联合培养项目。",
}

QUESTIONS = [
    ("Q01", "multi", "《大模型通识课》授课教师所在学院的行政负责人是谁？", ["doc04", "doc06", "doc02"]),
    ("Q02", "multi", "AI 协会的指导教师主讲课程的课程代码是什么？", ["doc11", "doc04", "doc06"]),
    ("Q03", "multi", "认知计算全国重点实验室依托的学院成立于哪一年？", ["doc10", "doc02"]),
    ("Q04", "multi", "《知识图谱导论》授课教师的研究方向是什么？", ["doc07", "doc04"]),
    ("Q05", "multi", "机器人战队的指导教师的研究方向是什么？", ["doc11", "doc04"]),
    ("Q06", "fact", "云山大学的校训是什么？", ["doc01"]),
    ("Q07", "fact", "国家奖学金的金额是多少？", ["doc09"]),
    ("Q08", "fact", "图书馆每天几点到几点开放？", ["doc13"]),
    ("Q09", "fact", "人工智能学院成立于哪一年？", ["doc02"]),
    ("Q10", "fact", "春季学期什么时候开学？", ["doc12"]),
    ("Q11", "global", "云山大学有哪些科研平台？分别依托谁建设？", ["doc10", "doc02", "doc03"]),
    ("Q12", "global", "学校的学生奖励体系包含哪些项目？", ["doc09"]),
    ("Q13", "global", "人工智能学院有哪些教师？各自做什么研究方向？", ["doc04"]),
    ("Q14", "global", "本科生选课有哪些限制和先修要求？", ["doc08"]),
    ("Q15", "global", "对想深入学习大模型的学生，学校提供哪些课程和课外活动？", ["doc06", "doc11"]),
    ("Q16", "fact", "人工智能分馆位于哪栋楼的哪一层？", ["doc13"]),
    ("Q17", "fact", "校内校巴有几条线路？", ["doc14"]),
    ("Q18", "multi", "研究检索增强生成的教师指导哪个学生社团？", ["doc04", "doc11"]),
    ("Q19", "multi", "课程代码 AI3305 对应的课程需要先修什么？", ["doc07", "doc08"]),
    ("Q20", "global", "概述云山大学的所在地、春季开学日期和地铁到校方式。", ["doc01", "doc12", "doc14"]),
]

# 主检索基线固定使用指南40条三元组；真实API抽取另存补充实验，不能混称同一套图。
TRIPLES = [
    ("李文瀚", "任职于", "人工智能学院", "doc04"), ("李文瀚", "职称", "教授", "doc04"),
    ("李文瀚", "入职年份", "2019 年", "doc04"), ("李文瀚", "研究方向", "检索增强生成", "doc04"),
    ("李文瀚", "主讲", "大模型通识课", "doc06"), ("李文瀚", "指导", "AI 协会", "doc11"),
    ("赵婉晴", "任职于", "人工智能学院", "doc04"), ("赵婉晴", "研究方向", "知识图谱", "doc04"),
    ("赵婉晴", "主讲", "知识图谱导论", "doc07"), ("孙浩然", "任职于", "人工智能学院", "doc04"),
    ("孙浩然", "研究方向", "强化学习", "doc04"), ("孙浩然", "指导", "机器人战队", "doc11"),
    ("人工智能学院", "院长", "王启明", "doc02"), ("人工智能学院", "成立于", "2018 年", "doc02"),
    ("人工智能学院", "下设系", "机器学习系", "doc02"), ("人工智能学院", "下设系", "智能科学系", "doc02"),
    ("人工智能学院", "下设系", "认知计算系", "doc02"), ("认知计算全国重点实验室", "依托", "人工智能学院", "doc10"),
    ("岭南超算中心", "参与共建", "计算机学院", "doc10"), ("计算机学院", "院长", "陈国峰", "doc03"),
    ("计算机学院", "前身", "计算机系", "doc03"), ("周天宇", "任职于", "计算机学院", "doc05"),
    ("周天宇", "研究方向", "分布式系统", "doc05"), ("林小雨", "任职于", "计算机学院", "doc05"),
    ("林小雨", "研究方向", "数据库系统", "doc05"), ("大模型通识课", "课程代码", "AI2101", "doc06"),
    ("大模型通识课", "学分", "3 学分", "doc06"), ("大模型通识课", "开设学期", "春季学期", "doc06"),
    ("知识图谱导论", "课程代码", "AI3305", "doc07"), ("知识图谱导论", "先修课程", "数据结构", "doc08"),
    ("AI 协会", "指导教师", "李文瀚", "doc11"), ("机器人战队", "指导教师", "孙浩然", "doc11"),
    ("国家奖学金", "金额", "8000 元/年", "doc09"), ("校长奖学金", "金额", "20000 元/年", "doc09"),
    ("云山一等奖学金", "金额", "3000 元/年", "doc09"), ("云山大学", "校训", "格物致知", "doc01"),
    ("云山大学", "建校于", "1958 年", "doc01"), ("云山大学", "位于", "岭南省云州市", "doc01"),
    ("图书馆", "藏书量", "380 万册", "doc13"), ("春季学期", "开学日期", "3 月 2 日", "doc12"),
]

# 按指南的九个主题重写聚合；删去指南示例中没有原文依据的「每年 15 人」。
ENTRIES = [
    ("云山大学", "云山大学在岭南省云州市，1958 年建校，校训格物致知。春季学期 3 月 2 日开学；地铁 3 号线云大站距离东门 200 米，校内有五条校巴线路。", ["doc01", "doc12", "doc14"]),
    ("人工智能学院", "人工智能学院 2018 年成立，由王启明任院长，设机器学习、智能科学、认知计算三个系。认知计算全国重点实验室依托该院建设，该院还与南洋理工大学开展本科联合培养。", ["doc02", "doc10", "doc15"]),
    ("计算机学院", "计算机学院的前身是 1985 年成立的计算机系，现任院长陈国峰。学院设计算机科学与技术、软件工程专业，参与岭南超算中心运行管理。", ["doc03", "doc10"]),
    ("人工智能学院教师", "李文瀚研究检索增强生成，主讲大模型通识课并指导 AI 协会；赵婉晴研究知识图谱，主讲知识图谱导论；孙浩然研究强化学习并指导机器人战队。", ["doc04", "doc11"]),
    ("计算机学院教师", "周天宇是研究分布式系统的教授，林小雨是研究数据库系统的副教授。", ["doc05"]),
    ("大模型通识课", "大模型通识课代码 AI2101，春季开设，共 3 学分 32 学时，由李文瀚授课，面向全校本科生且无先修要求。", ["doc06"]),
    ("知识图谱导论", "知识图谱导论代码 AI3305，秋季开设，由赵婉晴授课，面向人工智能学院研究生，要求先修数据结构。", ["doc07", "doc08"]),
    ("奖学金体系", "学生奖励包括每年 8000 元的国家奖学金、20000 元的校长奖学金和 3000 元的云山一等奖学金。", ["doc09"]),
    ("学生社团与校园生活", "AI 协会由李文瀚指导，每周三晚研读论文；机器人战队由孙浩然指导，参加全国机器人大赛。图书馆藏书 380 万册，每天 7:00 至 22:00 开放，人工智能分馆位于理科楼 B 座 3 层。", ["doc11", "doc13"]),
]


def chunks(size=120, overlap=20):
    return [(doc, text[i:i + size]) for doc, text in CORPUS.items()
            for i in range(0, len(text), size - overlap)]


def doc_rank(items):
    """把块/三元组/条目命中按首次出现次序映射到唯一文档。"""
    result = []
    for doc, score, detail in items:
        if doc not in [x[0] for x in result]:
            result.append((doc, float(score), detail))
    return result


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL, cache_folder=str(OUT / "model_cache"),
                                         device="cpu", local_files_only=True)

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True,
                                            show_progress_bar=False))


class VectorRAG:
    name = "Vector RAG"

    def __init__(self, emb):
        self.chunks = chunks()
        self.vectors = emb.encode([text for _, text in self.chunks])
        self.emb = emb

    def retrieve(self, query):
        scores = self.vectors @ self.emb.encode([query])[0]
        order = np.argsort(-scores)
        return doc_rank([(self.chunks[i][0], scores[i], self.chunks[i][1]) for i in order])

    def hybrid(self, query):
        """选做：字符 bigram 与语义排名按 RRF(k=60) 融合。"""
        sem = self.vectors @ self.emb.encode([query])[0]
        bigrams = lambda s: {s[i:i + 2] for i in range(len(s) - 1)}
        qgrams = bigrams(query.replace(" ", ""))
        lexical = np.array([len(qgrams & bigrams(t.replace(" ", ""))) /
                            max(len(qgrams), 1) for _, t in self.chunks])
        srank = np.argsort(-sem)
        lrank = np.argsort(-lexical)
        rrf = np.zeros(len(self.chunks))
        for rank, i in enumerate(srank, 1):
            rrf[i] += 1 / (60 + rank)
        for rank, i in enumerate(lrank, 1):
            rrf[i] += 1 / (60 + rank)
        return doc_rank([(self.chunks[i][0], rrf[i], self.chunks[i][1])
                         for i in np.argsort(-rrf)])


class GraphRAG:
    name = "GraphRAG"

    def __init__(self, emb):
        self.emb = emb
        self.graph = nx.MultiDiGraph()
        for h, r, t, src in TRIPLES:
            self.graph.add_edge(h, t, relation=r, source=src)
        self.communities = list(nx.community.greedy_modularity_communities(
            nx.Graph(self.graph.to_undirected())))
        self.summaries = []
        for community in self.communities:
            edges = [(h, r, t, src) for h, r, t, src in TRIPLES
                     if h in community and t in community]
            summary = "；".join(f"{h}{r}{t}" for h, r, t, _ in edges)
            docs = list(dict.fromkeys(src for _, _, _, src in edges))
            self.summaries.append((summary, docs))
        self.summary_vectors = emb.encode([s for s, _ in self.summaries])
        self.triple_vectors = emb.encode([f"{h}{r}{t}" for h, r, t, _ in TRIPLES])

    def entities(self, query):
        # 显式记录唯一可证明的别名，不声称已做完整实体归一化。
        query = query.replace("云大", "云山大学")
        return sorted((n for n in self.graph if n in query), key=len, reverse=True)

    def retrieve(self, query, scope="local"):
        qv = self.emb.encode([query])[0]
        if scope == "global":
            scores = self.summary_vectors @ qv
            order = np.argsort(-scores)
            hits = [(doc, scores[i], self.summaries[i][0])
                    for i in order for doc in self.summaries[i][1]]
            return doc_rank(hits)
        seeds = self.entities(query)
        if not seeds:
            return self.retrieve(query, "global")
        queue = deque((n, 0) for n in seeds)
        visited = set(seeds)
        reached = set()
        while queue:
            node, depth = queue.popleft()
            if depth == 3:
                continue
            for neighbor in self.graph.to_undirected().neighbors(node):
                for h, r, t, src in TRIPLES:
                    if {h, t} == {node, neighbor}:
                        reached.add((h, r, t, src))
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        scored = []
        for i, (h, r, t, src) in enumerate(TRIPLES):
            if (h, r, t, src) in reached:
                scored.append((src, float(self.triple_vectors[i] @ qv), f"{h}—{r}→{t}"))
        return doc_rank(sorted(scored, key=lambda x: -x[1]))


class WikiRAG:
    name = "WikiRAG"

    def __init__(self, emb):
        self.emb = emb
        self.vectors = emb.encode([title + "。" + text for title, text, _ in ENTRIES])

    def retrieve(self, query):
        scores = self.vectors @ self.emb.encode([query])[0]
        return doc_rank([(doc, scores[i], ENTRIES[i][0])
                         for i in np.argsort(-scores) for doc in ENTRIES[i][2]])


def score(system, questions, k=5, scope="local"):
    details = []
    for qid, typ, query, evidence in questions:
        hits = system.retrieve(query, scope) if isinstance(system, GraphRAG) else system.retrieve(query)
        docs = [d for d, _, _ in hits[:k]]
        first = next((i for i, d in enumerate(docs, 1) if d in evidence), None)
        details.append({"system": system.name + (" global" if isinstance(system, GraphRAG) and scope == "global" else ""),
                        "id": qid, "type": typ, "question": query, "evidence": evidence,
                        "hits": docs, "rank": first, "recall": len(set(docs) & set(evidence)) / len(evidence),
                        "rr": 1 / first if first else 0, "full": set(evidence) <= set(docs),
                        "scored_hits": [{"doc": d, "score": round(s, 4), "detail": detail}
                                        for d, s, detail in hits[:k]]})
    return details


def aggregates(rows):
    result = []
    for system in dict.fromkeys(r["system"] for r in rows):
        for typ in ("fact", "multi", "global"):
            part = [r for r in rows if r["system"] == system and r["type"] == typ]
            result.append({"system": system, "type": typ, "n": len(part),
                           "recall": round(float(np.mean([r["recall"] for r in part])), 4),
                           "mrr": round(float(np.mean([r["rr"] for r in part])), 4),
                           "full": sum(r["full"] for r in part)})
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_graph_figure(graph):
    # 节点名是中文；图中以编号显示，编号映射保存在 graph_nodes.csv。
    nodes = list(graph.nodes)
    labels = {n: str(i + 1) for i, n in enumerate(nodes)}
    write_csv(OUT / "graph_nodes.csv", [{"number": i + 1, "entity": n}
                                       for i, n in enumerate(nodes)])
    fig, ax = plt.subplots(figsize=(13, 10))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(graph, pos, labels=labels, node_size=520, font_size=7,
                     node_color="#9bd0e8", edge_color="#a4a4a4", arrows=False, ax=ax)
    ax.set_title("GraphRAG entity graph (node index in graph_nodes.csv)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "graph.png", dpi=180)
    plt.close(fig)


def make_wiki_figure():
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc", size=9)
    title_font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc", size=11, weight="bold")
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (title, body, docs) in zip(axes.flat, ENTRIES):
        ax.set_facecolor("#edf7fb")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.04, 0.94, title, transform=ax.transAxes, va="top", fontproperties=title_font)
        ax.text(0.04, 0.78, "\n".join(textwrap.wrap(body, width=26)),
                transform=ax.transAxes, va="top", fontproperties=font, linespacing=1.5)
        ax.text(0.04, 0.04, " / ".join(docs), transform=ax.transAxes,
                va="bottom", fontsize=9, color="#345b70")
    fig.tight_layout()
    fig.savefig(OUT / "wiki_entries.png", dpi=160)
    plt.close(fig)


def make_result_figures(summary, scan, q01):
    names = ["Vector RAG", "GraphRAG", "WikiRAG"]
    types = ["fact", "multi", "global"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    for j, name in enumerate(names):
        vals = [next(r["recall"] for r in summary if r["system"] == name and r["type"] == typ)
                for typ in types]
        ax.bar(x + (j - 1) * 0.25, vals, width=0.24, label=name)
    ax.set_xticks(x, types)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Recall@5 (20 questions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "recall_20.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, typ in zip(axes, types):
        for name in names:
            values = [next(r["recall"] for r in scan if r["system"] == name and
                           r["type"] == typ and r["k"] == k) for k in (1, 2, 3, 5)]
            ax.plot((1, 2, 3, 5), values, marker="o", label=name)
        ax.set_title(typ)
        ax.set_xticks((1, 2, 3, 5))
        ax.set_xlabel("Top-K")
    axes[0].set_ylabel("Recall (built-in 15)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "k_sensitivity_15.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.axis("off")
    table = [[str(i), h["doc"], f'{h["score"]:.4f}',
              "yes" if h["doc"] in q01["evidence"] else "no"]
             for i, h in enumerate(q01["scored_hits"], 1)]
    ax.table(cellText=table, colLabels=["rank", "document", "cosine", "gold evidence"],
             loc="center", cellLoc="center", colColours=["#d9edf7"] * 4)
    ax.set_title("Vector RAG Q01: retrieved documents and scores")
    fig.tight_layout()
    fig.savefig(OUT / "vector_q01.png", dpi=180)
    plt.close(fig)


def run_all():
    OUT.mkdir(exist_ok=True)
    start = time.perf_counter()
    emb = Embedder()
    model_load_seconds = time.perf_counter() - start
    t0 = time.perf_counter()
    vector = VectorRAG(emb)
    vector_build_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    graph = GraphRAG(emb)
    graph_build_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    wiki = WikiRAG(emb)
    wiki_build_seconds = time.perf_counter() - t0
    systems = [vector, graph, wiki]
    rows = [r for s in systems for r in score(s, QUESTIONS)]
    rows += score(graph, QUESTIONS, scope="global")
    summary = aggregates(rows)
    scan = []
    for k in (1, 2, 3, 5):
        for s in systems:
            for row in aggregates(score(s, QUESTIONS[:15], k)):
                scan.append({"k": k, **row})
        for row in aggregates(score(graph, QUESTIONS[:15], k, "global")):
            scan.append({"k": k, **row})
    hybrid = aggregates([{**r, "system": "Vector+RRF"} for r in score_hybrid(vector, QUESTIONS)])
    write_csv(OUT / "metrics_20.csv", summary)
    write_csv(OUT / "k_scan_15.csv", scan)
    write_csv(OUT / "hybrid_metrics_20.csv", hybrid)
    write_csv(OUT / "questions.csv", [{"id": qid, "type": typ, "question": q, "evidence": ",".join(ev)}
                                       for qid, typ, q, ev in QUESTIONS])
    write_csv(OUT / "triples.csv", [{"head": h, "relation": r, "tail": t, "source": d}
                                     for h, r, t, d in TRIPLES])
    write_csv(OUT / "wiki_entries.csv", [{"title": title, "text": body, "source_docs": ",".join(docs),
                                          "length": len(body)} for title, body, docs in ENTRIES])
    (OUT / "retrieval_details.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    make_graph_figure(graph.graph)
    make_wiki_figure()
    make_result_figures(summary, scan, next(r for r in rows if r["system"] == "Vector RAG" and r["id"] == "Q01"))
    manual = [next(r for r in rows if r["system"] == "WikiRAG" and r["id"] == qid)
              for qid in ("Q01", "Q08", "Q14")]
    write_csv(OUT / "mrr_manual_3.csv", [{"system": r["system"], "id": r["id"],
                                          "type": r["type"], "top5": ",".join(r["hits"]),
                                          "evidence": ",".join(r["evidence"]),
                                          "first_hit_rank": r["rank"], "reciprocal_rank": r["rr"]}
                                         for r in manual])
    mrr = sum(r["rr"] for r in manual) / 3
    (OUT / "mrr_manual_3.txt").write_text(
        f"WikiRAG Q01/Q08/Q14 首次命中位置：{[r['rank'] for r in manual]}\n"
        f"MRR = ({' + '.join('1/' + str(r['rank']) for r in manual)}) / 3 = {mrr:.4f}\n",
        encoding="utf-8")
    write_manual_review(rows)
    config = {"model": MODEL, "device": "cpu", "chunk_size": 120, "overlap": 20,
              "os": platform.platform(), "python": platform.python_version(),
              "torch": torch.__version__, "numpy": np.__version__,
              "sklearn": sklearn.__version__, "networkx": nx.__version__,
              "matplotlib": matplotlib.__version__,
              "sentence_transformers": sentence_transformers.__version__,
              "default_k": 5, "docs": len(CORPUS), "chunks": len(vector.chunks),
              "triples": len(TRIPLES), "graph_nodes": graph.graph.number_of_nodes(),
              "graph_edges": graph.graph.number_of_edges(), "communities": len(graph.communities),
              "wiki_entries": len(ENTRIES), "wiki_average_chars": round(np.mean([len(t) for _, t, _ in ENTRIES]), 1),
              "model_load_seconds": round(model_load_seconds, 2),
              "vector_build_seconds": round(vector_build_seconds, 2),
              "graph_build_seconds": round(graph_build_seconds, 2),
              "wiki_build_seconds": round(wiki_build_seconds, 2),
              "total_run_seconds": round(time.perf_counter() - start, 2),
              "llm_api": "未使用；三元组和条目由指南预置并检查改写"}
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("\n20 题 Recall@5 / MRR：")
    for row in summary:
        print(row)
    print("\n选做 RRF：")
    for row in hybrid:
        print(row)
    print("\n结果在：", OUT)


def score_hybrid(vector, questions):
    result = []
    for qid, typ, q, evidence in questions:
        hits = vector.hybrid(q)[:5]
        docs = [d for d, _, _ in hits]
        first = next((i for i, d in enumerate(docs, 1) if d in evidence), None)
        result.append({"system": "Vector+RRF", "id": qid, "type": typ,
                       "recall": len(set(docs) & set(evidence)) / len(evidence),
                       "rr": 1 / first if first else 0, "full": set(evidence) <= set(docs)})
    return result


def write_manual_review(rows):
    """无 LLM API：按实际检索上下文写可核对的示范回答，不伪称模型生成。"""
    answers = {
        ("Vector RAG", "Q01"): ("王启明教授。", "doc06,doc04,doc02"),
        ("Vector RAG", "Q06"): ("校训是格物致知。", "doc01"),
        ("Vector RAG", "Q11"): ("有认知计算全国重点实验室，依托人工智能学院；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理。", "doc10"),
        ("GraphRAG", "Q01"): ("王启明教授。", "doc06,doc04,doc02"),
        ("GraphRAG", "Q06"): ("校训是格物致知。", "doc01"),
        ("GraphRAG", "Q11"): ("检索到的资料不足以确定科研平台。", ""),
        ("WikiRAG", "Q01"): ("检索到的资料未给出学院行政负责人，无法确认。", ""),
        ("WikiRAG", "Q06"): ("校训是格物致知。", "doc01"),
        ("WikiRAG", "Q11"): ("检索到的资料不足以列举科研平台。", ""),
    }
    review = []
    for (system, qid), (answer, cited) in answers.items():
        detail = next(r for r in rows if r["system"] == system and r["id"] == qid)
        assert set(filter(None, cited.split(","))) <= set(detail["hits"])
        review.append({"system": system, "id": qid, "type": detail["type"],
                       "retrieved": ",".join(detail["hits"]), "answer_excerpt": answer,
                       "cited_docs": cited, "unsupported_claim": "无",
                       "codex_initial_faithfulness_1to5": 5,
                       "note": "CODEX 根据检索结果撰写并初评；非 LLM 生成，需本人复核。"})
    write_csv(OUT / "faithfulness_9_review.csv", review)

    selected = [("Vector RAG", "Q01", 2, "桥接 doc02 排第 4；K=2 的窗口装不下完整关系链"),
                ("Vector RAG", "Q11", 5, "doc02 未入 Top-5；指南金标准同时要求机构背景文档"),
                ("GraphRAG", "Q11", 5, "问题只匹配云山大学节点，局部三跳未连到两个科研平台"),
                ("GraphRAG", "Q16", 5, "预置三元组没有人工智能分馆位置"),
                ("WikiRAG", "Q01", 5, "课程和教师条目已检出，但学院条目来源 doc02 被挤出 Top-5"),
                ("WikiRAG", "Q11", 5, "科研平台相关条目排序落后，前五个映射文档都不是标注证据")]
    cases = []
    for system, qid, k, cause in selected:
        r = next(x for x in rows if x["system"] == system and x["id"] == qid)
        docs = r["hits"][:k]
        cases.append({"system": system, "id": qid, "k": k,
                      "evidence": ",".join(r["evidence"]), "retrieved": ",".join(docs),
                      "missing": ",".join(d for d in r["evidence"] if d not in docs),
                      "cause": cause})
    write_csv(OUT / "failure_cases.csv", cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "vector", "graph", "wiki", "hybrid"], default="all")
    parser.add_argument("--q", default="Q01")
    parser.add_argument("--scope", choices=["local", "global"], default="local")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="用环境变量中的 API 对当前检索原文生成回答")
    parser.add_argument("--mrr_check", help="例如 3,1,2，输出手算式")
    args = parser.parse_args()
    if args.mrr_check:
        ranks = [int(x) for x in args.mrr_check.split(",")]
        print(f"MRR = ({' + '.join(f'1/{r}' for r in ranks)}) / {len(ranks)} = "
              f"{sum(1 / r for r in ranks) / len(ranks):.4f}")
        return
    if args.mode == "all":
        if args.generate:
            parser.error("--generate 请搭配 vector、graph、wiki 或 hybrid 单题模式")
        run_all()
        return
    emb = Embedder()
    system = {"vector": VectorRAG, "graph": GraphRAG, "wiki": WikiRAG,
              "hybrid": VectorRAG}[args.mode](emb)
    qid, typ, q, evidence = next(item for item in QUESTIONS if item[0] == args.q.upper())
    hits = (system.hybrid(q) if args.mode == "hybrid" else
            system.retrieve(q, args.scope) if args.mode == "graph" else system.retrieve(q))
    print(f"{qid} [{typ}] {q}\n标准证据：{evidence}")
    for i, (doc, similarity, detail) in enumerate(hits[:args.k], 1):
        print(f"{i}. {doc}  score={similarity:.4f}  命中={doc in evidence}  线索={detail}\n   原文：{CORPUS[doc]}")
    got = {d for d, _, _ in hits[:args.k]}
    print(f"Recall@{args.k}={len(got & set(evidence)) / len(evidence):.3f}")
    if args.generate:
        context = "\n\n".join(f"[{doc}] {CORPUS[doc]}" for doc, _, _ in hits[:args.k])
        result = llm_chat(ANSWER_PROMPT.format(q=q, ctx=context), temperature=0)
        print(f"LLM（{result['model']}）真实回答：{result['raw_output']}")


if __name__ == "__main__":
    main()
