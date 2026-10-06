# -*- coding: utf-8 -*-
"""作业四第二层：LLM 自动知识构建与生成；复用固定基准的检索记录。

凭据只读环境变量 LLM_API_KEY。默认结果另存；已完成调用直接复用，避免重复计费。
不依赖新增软件，使用 Python 标准库连接 DeepSeek 官方接口。
"""
import argparse
import ast
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
BASELINE = ROOT / "lab04_outputs"
DEFAULT_OUT = BASELINE / "api_20261005"

V1 = ('从下面这篇大学介绍文档中抽取「头实体-关系-尾实体」三元组，只输出 JSON 数组，'
      '不要输出任何其他文字，格式如 [["李文瀚","任职于","人工智能学院"]]。\n'
      '关系用 2~4 字短语（任职于/研究方向/主讲/成立于 等），一篇文档抽 5~10 条。')
V2 = V1 + ('\n仅抽取文档明确陈述的事实，不推断、不补充常识。使用完整实体名称；'
           '每条表示一个原子事实，关系名称稳定一致；删除完全重复三元组。'
           '输出必须是合法 JSON，三个元素均为非空字符串。事实不足 5 条时按实际数量输出，'
           '事实较多时优先保留任职、研究、课程、平台及位置等主要事实，最多 10 条。')
V3 = V2 + ('\n特别区分依托、共建、参与运行管理、任职于、主讲、指导，'
           '不可将参与运行管理改写为共建，不可混淆关系方向。关系语义准确优先于 2~4 字长度。'
           '仅归一化原文明确可确认的别名，不做模糊实体合并；保留原文的人名、机构全称、'
           '年份、数量、地点限定词，不捏造职务或身份。输出前逐条核对头实体、关系、'
           '尾实体是否都有当前文档依据；无依据则删除，不以凑数量为理由补造事实。')
EXTRACT_PROMPTS = {"V1": V1, "V2": V2, "V3": V3}
ANSWER_PROMPT = ('你是问答助手。只依据下面资料回答问题；资料不足以回答时明确说'
                 '「资料中未提及」，禁止编造。不要把未提及解释为不存在。\n'
                 '问题：{q}\n资料：\n{ctx}\n'
                 '用 1~3 句话回答，句末用括号注明用到的资料编号。')


def source_constants():
    """直接读取原脚本常量，既不加载 embedding，也不运行旧结果写入函数。"""
    tree = ast.parse((CODE_DIR / "hw4_rag.py").read_text(encoding="utf-8-sig"))
    names = {"CORPUS", "QUESTIONS", "TRIPLES", "ENTRIES"}
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    return values


def llm_chat(prompt, temperature=0):
    """单次调用，不隐藏重试；错误只显示状态码，避免凭据进入日志。"""
    key = os.environ["LLM_API_KEY"].strip()
    model = os.environ.get("LLM_MODEL", "deepseek-flash")
    base = os.environ.get("LLM_API_BASE", os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")).rstrip("/")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "max_tokens": 2048,
               "thinking": {"type": "disabled"}}
    request = urllib.request.Request(base + "/chat/completions",
                                    data=json.dumps(payload).encode("utf-8"),
                                    headers={"Authorization": "Bearer " + key,
                                             "Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"API HTTP {error.code}，停止调用，请检查服务状态或账户") from None
    except urllib.error.URLError:
        raise RuntimeError("API 网络连接失败，停止调用") from None
    choice = result["choices"][0]
    return {"provider": "DeepSeek", "base_url": base, "requested_model": model,
            "model": result["model"], "temperature": temperature,
            "thinking": "disabled", "max_tokens": 2048,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "response_id": result.get("id"), "usage": result.get("usage"),
            "finish_reason": choice.get("finish_reason"),
            "raw_output": choice["message"]["content"], "prompt": prompt}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def cached_call(path, prompt, metadata):
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["prompt"] != prompt:
            raise ValueError("已有提示词不同，请使用新的 --output 目录，不能覆盖历史调用")
        print("复用已保存调用", path.name, flush=True)
        return result
    result = {**llm_chat(prompt), **metadata}
    write_json(path, result)
    print("已保存", path.name, result["model"], result["elapsed_seconds"], "s", flush=True)
    return result


def parse_triples(result):
    # 严格解析完整输出，代码围栏或多余解释都计为 JSON 不符合要求，不悄悄修复。
    try:
        triples = json.loads(result["raw_output"])
    except json.JSONDecodeError:
        return [], False, False
    schema_ok = isinstance(triples, list) and all(
        isinstance(x, list) and len(x) == 3 and all(isinstance(y, str) and y.strip() for y in x)
        for x in triples)
    # 保留合法项，另行标记整篇的 schema 失败；不能因为一个空尾实体丢掉所有有效项。
    valid = [x for x in triples if isinstance(x, list) and len(x) == 3
             and all(isinstance(y, str) and y.strip() for y in x)] if isinstance(triples, list) else []
    return valid, True, schema_ok


def extract_doc(out, phase, version, doc_id, text):
    prompt = EXTRACT_PROMPTS[version] + "\n文档编号：" + doc_id + "\n文档：\n" + text
    result = cached_call(out / phase / f"{version}_{doc_id}.json", prompt,
                         {"phase": phase, "version": version, "doc_id": doc_id, "source_text": text})
    triples, json_ok, schema_ok = parse_triples(result)
    raw_parsed = json.loads(result["raw_output"]) if json_ok else None
    raw_count = len(raw_parsed) if isinstance(raw_parsed, list) else 0
    return {**result, "triples": triples, "json_ok": json_ok, "schema_ok": schema_ok,
            "triple_count": len(triples),
            "raw_triple_count": raw_count,
            "rejected_item_count": raw_count - len(triples),
            "duplicate_count": len(triples) - len({tuple(t) for t in triples})}


def extraction_phase(out, phase, versions, docs):
    corpus = source_constants()["CORPUS"]
    results = [extract_doc(out, phase, version, doc, corpus[doc])
               for version in versions for doc in docs]
    write_json(out / f"{phase}_results.json", results)
    fields = ["version", "doc_id", "model", "json_ok", "schema_ok", "raw_triple_count", "triple_count", "rejected_item_count",
              "duplicate_count", "elapsed_seconds"]
    with (out / f"{phase}_statistics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in results)
    if phase == "full_extract":
        sourced = [{"head": h, "relation": rel, "tail": t, "source": r["doc_id"]}
                   for r in results for h, rel, t in r["triples"]]
        write_json(out / "llm_triples_with_sources.json", sourced)
        write_json(out / "full_extract_summary.json", {
            "version": versions[0], "documents": len(results), "api_successes": len(results),
            "api_failures": 0, "json_failures": sum(not r["json_ok"] for r in results),
            "schema_failures": sum(not r["schema_ok"] for r in results),
            "raw_triples": sum(r["raw_triple_count"] for r in results),
            "rejected_items": sum(r["rejected_item_count"] for r in results),
            "triples": len(sourced), "elapsed_seconds": round(sum(r["elapsed_seconds"] for r in results), 3),
            "human_review_status": "pending", "human_review_seconds": None,
            "baseline_triples_unchanged": True})
        write_json(out / "llm_triples_rejected.json", [
            {"source": r["doc_id"], "triple": x, "reason": "三字符串结构不合格或存在空实体"}
            for r in results if r["json_ok"] for x in json.loads(r["raw_output"])
            if not (isinstance(x, list) and len(x) == 3 and all(isinstance(y, str) and y.strip() for y in x))])
    return results


def generate_answers(out):
    corpus = source_constants()["CORPUS"]
    retrievals = json.loads((BASELINE / "retrieval_details.json").read_text(encoding="utf-8"))
    results = []
    for system in ["Vector RAG", "GraphRAG", "WikiRAG"]:
        for qid in ["Q01", "Q06", "Q11"]:
            item = next(x for x in retrievals if x["system"] == system and x["id"] == qid)
            docs = item["hits"][:5]
            context = "\n\n".join(f"[{d}] {corpus[d]}" for d in docs)
            prompt = ANSWER_PROMPT.format(q=item["question"], ctx=context)
            name = {"Vector RAG": "vector", "GraphRAG": "graph_local", "WikiRAG": "wiki"}[system]
            r = cached_call(out / "answers" / f"{name}_{qid}.json", prompt,
                            {"system": system, "scope": "local" if system == "GraphRAG" else None,
                             "qid": qid, "question": item["question"], "doc_ids": docs,
                             "context": context, "evidence": item["evidence"], "top_k": 5,
                             "retrieval_source": "lab04_outputs/retrieval_details.json"})
            results.append(r)
    write_json(out / "answers_9.json", results)
    review_path = out / "faithfulness_9_human_review.csv"
    if not review_path.exists():
        with review_path.open("w", newline="", encoding="utf-8-sig") as handle:
            fields = ["system", "qid", "doc_ids", "answer", "human_score", "human_notes", "review_status"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({"system": r["system"], "qid": r["qid"], "doc_ids": ",".join(r["doc_ids"]),
                              "answer": r["raw_output"], "human_score": "", "human_notes": "",
                              "review_status": "pending"} for r in results)
    review_md = out / "人工复核材料.md"
    if not review_md.exists():
        parts = ["# 九个真实 API 回答的人工复核材料\n",
                 "请逐句核对下列原始上下文，填写同目录 CSV 的 human_score（1～5）、human_notes 和 review_status。忠实度与回答完整性分别判断；AI 没有代填人工评分。\n"]
        for r in results:
            parts += [f"## {r['system']} {r['qid']}\n", f"问题：{r['question']}\n",
                      "回答：" + r["raw_output"] + "\n", "检索原文：\n" + r["context"] + "\n"]
        review_md.write_text("\n".join(parts), encoding="utf-8")
    return results


def community_summaries(out):
    """补充实验：仍是固定40条图谱，只有 Global 排序用的摘要文本改变。"""
    import networkx as nx
    values = source_constants()
    triples, corpus = values["TRIPLES"], values["CORPUS"]
    graph = nx.MultiDiGraph()
    for h, rel, t, src in triples:
        graph.add_edge(h, t, relation=rel, source=src)
    communities = list(nx.community.greedy_modularity_communities(nx.Graph(graph.to_undirected())))
    results = []
    for i, community in enumerate(communities, 1):
        edges = [x for x in triples if x[0] in community and x[2] in community]
        docs = list(dict.fromkeys(x[3] for x in edges))
        prompt = ('依据下面社区的节点、关系和来源原文写一段简洁社区摘要，只输出摘要正文。'
                  '禁止补充常识、无依据数字或身份；如预置关系与来源原文冲突，以原文为准，'
                  '不得把参与运行管理写成共建。不要评论检索效果。\n节点：'
                  + json.dumps(sorted(community), ensure_ascii=False) + '\n关系及来源：'
                  + json.dumps(edges, ensure_ascii=False) + '\n来源原文：\n'
                  + '\n'.join(f'[{d}] {corpus[d]}' for d in docs))
        results.append(cached_call(out / 'communities' / f'community_{i:02}.json', prompt,
                                  {'community_id': i, 'nodes': sorted(community),
                                   'edges': edges, 'doc_ids': docs, 'experiment': 'fixed40_llm_summary_supplement'}))
    write_json(out / 'llm_community_summaries.json', results)
    return results


def wiki_rewrites(out):
    values = source_constants()
    results = []
    for i, (title, original, docs) in enumerate(values['ENTRIES'], 1):
        prompt = ('仅依据提供的来源原文，为指定主题重写一个知识条目，只输出正文。'
                  '一个条目只围绕指定主题，跨文档聚合相关事实；保留明确年份、数量、身份及关系。'
                  '通常100～300字，但原文信息不足时宁可短于100字，不得凑字数或编造。'
                  '不添加主题外事实，不输出标题或解释。\n主题：' + title + '\n来源原文：\n'
                  + '\n'.join(f'[{d}] {values["CORPUS"][d]}' for d in docs))
        r = cached_call(out / 'wiki_rewrites' / f'entry_{i:02}.json', prompt,
                        {'entry_id': i, 'title': title, 'doc_ids': docs,
                         'original_body': original, 'experiment': 'wiki_llm_rewrite_supplement'})
        results.append({**r, 'body_length': len(r['raw_output'])})
    write_json(out / 'wiki_llm_rewrites.json', results)
    return results


def evaluate_supplements(out):
    """只写新目录；不调用原 run_all，不替换任何主指标文件。"""
    import hw4_rag as base
    embedding = base.Embedder()
    details = []
    if (out / 'llm_community_summaries.json').exists():
        rows = json.loads((out / 'llm_community_summaries.json').read_text(encoding='utf-8'))
        graph = base.GraphRAG(embedding)
        graph.name = 'GraphRAG fixed40 LLM-summary'
        graph.summaries = [(r['raw_output'], r['doc_ids']) for r in rows]
        graph.summary_vectors = embedding.encode([r['raw_output'] for r in rows])
        details.extend(base.score(graph, base.QUESTIONS, scope='global'))
    if (out / 'wiki_llm_rewrites.json').exists():
        rows = json.loads((out / 'wiki_llm_rewrites.json').read_text(encoding='utf-8'))
        wiki = base.WikiRAG(embedding)
        wiki.name = 'WikiRAG LLM-rewrite'
        wiki.vectors = embedding.encode([r['title'] + '。' + r['raw_output'] for r in rows])
        details.extend(base.score(wiki, base.QUESTIONS))
    if (out / 'llm_triples_with_sources.json').exists():
        rows = json.loads((out / 'llm_triples_with_sources.json').read_text(encoding='utf-8'))
        baseline_triples = base.TRIPLES
        try:
            # 仅当前补充进程的内存临时切换，原脚本和原输出始终不改。
            base.TRIPLES = [(r['head'], r['relation'], r['tail'], r['source']) for r in rows]
            graph = base.GraphRAG(embedding)
            graph.name = 'GraphRAG LLM-extracted uncorrected'
            details.extend(base.score(graph, base.QUESTIONS, scope='local'))
            details.extend(base.score(graph, base.QUESTIONS, scope='global'))
            write_json(out / 'llm_extracted_graph_config.json', {
                'triples': len(rows), 'nodes': len(graph.graph), 'communities': len(graph.communities),
                'schema_invalid_items_excluded': 2, 'human_correction': False,
                'extra_entity_normalization': False, 'local_depth': 3, 'top_k': 5,
                'purpose': '未经人工修正的API抽取补充，不替换固定40条主基线'})
        finally:
            base.TRIPLES = baseline_triples
    write_json(out / 'supplement_retrieval_details.json', details)
    base.write_csv(out / 'supplement_metrics_20.csv', base.aggregates(details))
    print('补充评测完成；原主指标未写入。', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["probe", "iterate", "extract", "generate", "community", "wiki", "evaluate"], required=True)
    parser.add_argument("--version", choices=list(EXTRACT_PROMPTS), default="V3")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        parser.error("输出必须位于当前作业仓库内")
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "probe":
        cached_call(out / "api_probe.json", "连通性测试：只回答‘API 可用’。", {"phase": "probe"})
    elif args.phase == "iterate":
        extraction_phase(out, "prompt_iterations", list(EXTRACT_PROMPTS), ["doc04", "doc10", "doc13"])
    elif args.phase == "extract":
        extraction_phase(out, "full_extract", [args.version], list(source_constants()["CORPUS"]))
    elif args.phase == "generate":
        generate_answers(out)
    elif args.phase == "community":
        community_summaries(out)
    elif args.phase == "wiki":
        wiki_rewrites(out)
    elif args.phase == "evaluate":
        evaluate_supplements(out)


if __name__ == "__main__":
    main()
