
import os
import json
import argparse
import re
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import time

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, InputExample, losses

from utils.graph_utils import search_paths
from utils import graph_utils
from utils.qa_utils import eval_answer_result

# =========================
# IO
# =========================
def ensure_parent_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)

# =========================
# Path statistics
# =========================
def _count_bucket_sizes(
    buckets: Dict[int, List[List[Tuple[str, str, str]]]],
    max_depth: int,
) -> Dict[int, int]:
    return {d: len(buckets.get(d, [])) for d in range(1, max_depth + 1)}


def _summarize_path_counts(values: List[int]) -> Dict[str, Any]:
    arr = np.asarray([int(v) for v in values], dtype=np.int64)
    if arr.size == 0:
        return {
            "num_samples": 0,
            "nonzero_samples": 0,
            "sum": 0,
            "max": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
        }
    return {
        "num_samples": int(arr.size),
        "nonzero_samples": int(np.count_nonzero(arr)),
        "sum": int(arr.sum()),
        "max": int(arr.max()),
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def _update_stage_stats(stage_stats: Dict[str, List[int]], stage: str, value: int):
    stage_stats[stage].append(int(value))


def _dump_stage_stats(stage_stats: Dict[str, List[int]], json_path: str, txt_path: str):
    summary = {k: _summarize_path_counts(v) for k, v in sorted(stage_stats.items())}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for stage, s in summary.items():
            f.write(f"[{stage}]\n")
            f.write(f"num_samples: {s['num_samples']}\n")
            f.write(f"nonzero_samples: {s['nonzero_samples']}\n")
            f.write(f"sum: {s['sum']}\n")
            f.write(f"max: {s['max']}\n")
            f.write(f"min: {s['min']}\n")
            f.write(f"mean: {s['mean']:.4f}\n")
            f.write(f"median: {s['median']:.4f}\n\n")

def _as_str_set(v: Any) -> Set[str]:
    if v is None:
        return set()
    if not isinstance(v, list):
        v = [v]
    return {s for s in (str(x).strip() for x in v) if s}

def _get_allowed_tails_from_record(it: dict, *, include_ground: bool) -> Set[str]:
    allowed = _as_str_set(it.get("prediction"))
    return allowed | _as_str_set(it.get("ground_tail_types")) if include_ground else allowed

_SPLIT_PAT = re.compile(r"\s*(?:->|→|—>|=>)\s*")
_WORD_SPLIT = re.compile(r"[a-zA-Z0-9]+")


def tokenize_arrow(s: str) -> List[str]:
    toks = _SPLIT_PAT.split(str(s).strip())
    return [t.strip() for t in toks if t.strip()]


def normalize_arrow_join(tokens_or_str: Any) -> str:
    if isinstance(tokens_or_str, str):
        toks = tokenize_arrow(tokens_or_str)
        return "->".join(toks) if toks else tokens_or_str.strip()
    if isinstance(tokens_or_str, (list, tuple)):
        toks = [str(x).strip() for x in tokens_or_str if str(x).strip()]
        return "->".join(toks) if toks else ""
    return str(tokens_or_str).strip()

# =========================
# Path / output formatting
# =========================
def _triple_path_to_str(triple_path: List[Tuple[Any, Any, Any]]) -> str:
    """[(h,r,t), ...] -> 'h->r1->t1->r2->t2'"""
    if not triple_path:
        return ""
    s = [str(triple_path[0][0])]
    for _, r, t in triple_path:
        s.append(str(r))
        s.append(str(t))
    return "->".join(s)

def _token_to_similarity_text(tok: str) -> str:
    """
    Convert path tokens into text for similarity computation.
    """
    tok = str(tok or "").strip()
    if not tok:
        return ""

    if "." in tok:
        return tok.replace(".", " ").replace("_", " ").strip()

    return tok


def _triple_path_to_similarity_text(triple_path: List[Tuple[str, str, str]]) -> str:
    """
    Use the full triple path for similarity computation.
    """
    if not triple_path:
        return ""

    parts = [str(triple_path[0][0]).strip()]
    for _, r, t in triple_path:
        parts.append(str(r).strip())
        parts.append(str(t).strip())

    text_parts = []
    for p in parts:
        p2 = _token_to_similarity_text(p)
        if p2:
            text_parts.append(p2)

    return " ; ".join(text_parts).strip()

def _rerank_triple_paths_by_similarity(
    question: str,
    triple_paths: List[List[Tuple[str, str, str]]],
    rank_sbert: Optional[SentenceTransformer],
    batch_size: int = 256,
) -> Tuple[List[List[Tuple[str, str, str]]], List[float]]:
    """
    Use Sentence-BERT to compute similarities between the question and triple paths.
    """
    triple_paths = _dedup_triple_paths(triple_paths)
    if not triple_paths:
        return [], []

    if rank_sbert is None:
        return triple_paths, [0.0] * len(triple_paths)

    texts = []
    for p in triple_paths:
        txt = _triple_path_to_similarity_text(p)
        if not txt:
            txt = _triple_path_to_str(p).replace("->", " ")
        texts.append(txt)

    q_emb = rank_sbert.encode(
        [str(question or "")],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    p_emb = rank_sbert.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sims = np.dot(p_emb, q_emb[0]).astype(np.float32)
    order = np.argsort(-sims, kind="stable")

    sorted_paths = [triple_paths[i] for i in order]
    sorted_scores = [float(sims[i]) for i in order]
    return sorted_paths, sorted_scores

# =========================
# Candidate TYPE PATH extraction
# =========================
def extract_candidate_type_paths_with_uniform_score(
    filtered_type_paths: List[Any]
) -> Tuple[List[str], Dict[str, float]]:
    if not filtered_type_paths:
        return [], {}

    seen: Set[str] = set()
    ordered: List[str] = []
    for tp in filtered_type_paths:
        s = normalize_arrow_join(tp)
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    tp2score = {s: 1.0 for s in ordered}
    return ordered, tp2score

# =========================
# Tail-type + Ontology last-hop mapping
# =========================
def load_tailtypes_jsonl(path: str, *, include_ground_tail_types: bool = False) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
            except Exception:
                continue
            qid = str(it.get("id", "") or it.get("qid", "") or "")
            if not qid:
                continue
            tails = _get_allowed_tails_from_record(it, include_ground=include_ground_tail_types)
            if tails:
                out[qid] = set(str(x).strip() for x in tails if str(x).strip())
    return out


def load_ontology_triples(path: str) -> List[Tuple[str, str, str]]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    triples: List[Tuple[str, str, str]] = []
    for x in (data or []):
        if isinstance(x, (list, tuple)) and len(x) >= 3:
            h, r, t = str(x[0]).strip(), str(x[1]).strip(), str(x[2]).strip()
            if h and r and t:
                triples.append((h, r, t))
    return triples


def build_tailtype2lastrels(ontology_triples: List[Tuple[str, str, str]]) -> Dict[str, Set[str]]:
    """
    tail_type -> set(last_rel)
    """
    mp: Dict[str, Set[str]] = defaultdict(set)
    for h, r, t in (ontology_triples or []):
        mp[t].add(r)
    return mp


def build_rel2heads_for_rels(graph_triples: List[Any], rels: Set[str]) -> Dict[str, Set[str]]:
    """
    Extract heads for the specified relations from sample.graph: rel -> {head}.
    """
    rel2heads: Dict[str, Set[str]] = defaultdict(set)
    if not rels:
        return rel2heads
    rels = set(str(r).strip() for r in rels if str(r).strip())
    for h, r, t in (graph_triples or []):
        r = str(r)
        if r in rels:
            rel2heads[r].add(str(h))
    return rel2heads


# =========================
# Bidirectional retrieval (forward prefixes meet backward last-hop heads)
# =========================
def _enumerate_prefix_paths(
    start_entities: List[str],
    kg_index: Dict[str, Dict[str, Set[str]]],
    max_depth: int,
    forbid_repeat_node: bool = True,
) -> Dict[int, List[List[Tuple[str, str, str]]]]:
    """
    Enumerate entity-level prefix paths from start_entities and bucket them by depth.
    """
    buckets: Dict[int, List[List[Tuple[str, str, str]]]] = defaultdict(list)
    if not start_entities or max_depth <= 0:
        return buckets

    # depth=1
    cur_paths: List[List[Tuple[str, str, str]]] = []
    for s in start_entities:
        s = str(s)
        if s not in kg_index:
            continue
        for r, tails in kg_index[s].items():
            for t in tails:
                cur_paths.append([(s, str(r), str(t))])

    if cur_paths:
        buckets[1].extend(cur_paths)

    # depth>=2
    for d in range(2, max_depth + 1):
        nxt: List[List[Tuple[str, str, str]]] = []
        for path in buckets[d - 1]:
            last_tail = path[-1][2]
            if last_tail not in kg_index:
                continue

            visited = set()
            if forbid_repeat_node:
                visited.add(path[0][0])
                for _, _, tt in path:
                    visited.add(tt)

            for r, tails in kg_index[last_tail].items():
                for t in tails:
                    t = str(t)
                    if forbid_repeat_node and t in visited:
                        continue
                    nxt.append(path + [(last_tail, str(r), t)])

        if not nxt:
            break
        buckets[d].extend(nxt)

    return buckets


def bidirectional_retrieve_reasoning_paths(
    q_entities: List[str],
    tail_types: Set[str],
    tailtype2lastrels: Dict[str, Set[str]],
    kg_index: Dict[str, Dict[str, Set[str]]],
    global_rel2heads: Dict[str, Set[str]],
    *,
    max_forward_depth: int = 3,
    max_reasoning_paths: int = 5000,
    keep_max_hops: int = 4,
    return_detail: bool = False,
):
    detail = {
        "tail_type_count": len(tail_types or set()),
        "candidate_last_rel_count": 0,
        "valid_last_rel_count": 0,
        "prefix_depth_counts": {d: 0 for d in range(1, max_forward_depth + 1)},
        "hop_counts": {h: 0 for h in range(1, 5)},
        "returned_count": 0,
        "selected_stage": "none",
    }

    keep_max_hops = int(max(1, keep_max_hops))
    keep_max_hops = min(keep_max_hops, max_forward_depth + 1)

    last_rels: Set[str] = set()
    for tt in (tail_types or set()):
        for r in (tailtype2lastrels.get(tt, set()) or set()):
            r = str(r).strip()
            if r:
                last_rels.add(r)
    if not last_rels:
        return ([], detail) if return_detail else []

    detail["candidate_last_rel_count"] = len(last_rels)
    valid_last_rels = {r for r in last_rels if global_rel2heads.get(r)}
    detail["valid_last_rel_count"] = len(valid_last_rels)

    if not valid_last_rels:
        return ([], detail) if return_detail else []

    prefix_buckets = _enumerate_prefix_paths(
        start_entities=q_entities,
        kg_index=kg_index,
        max_depth=max_forward_depth,
        forbid_repeat_node=True,
    )
    detail["prefix_depth_counts"] = _count_bucket_sizes(prefix_buckets, max_forward_depth)

    paths_by_hop: Dict[int, List[List[Tuple[str, str, str]]]] = {
        h: [] for h in range(1, keep_max_hops + 1)
    }
    total_added = 0

    def _try_add(hop: int, p: List[Tuple[str, str, str]]) -> bool:
        nonlocal total_added
        if hop < 1 or hop > keep_max_hops:
            return True
        if total_added >= max_reasoning_paths:
            return False
        paths_by_hop[hop].append(p)
        total_added += 1
        return total_added < max_reasoning_paths

    if 1 <= keep_max_hops:
        for qe in (q_entities or []):
            qe = str(qe)
            if qe not in kg_index:
                continue
            for lr in valid_last_rels:
                if qe not in global_rel2heads.get(lr, set()):
                    continue
                for tail in kg_index[qe].get(lr, set()):
                    if not _try_add(1, [(qe, lr, str(tail))]):
                        break
                if total_added >= max_reasoning_paths:
                    break
            if total_added >= max_reasoning_paths:
                break

    for d in range(1, max_forward_depth + 1):
        hop = d + 1
        if hop > keep_max_hops:
            break
        if total_added >= max_reasoning_paths:
            break

        for pref in prefix_buckets.get(d, []):
            if total_added >= max_reasoning_paths:
                break
            meet = str(pref[-1][2])

            for lr in valid_last_rels:
                if meet not in global_rel2heads.get(lr, set()):
                    continue
                if meet in kg_index and lr in kg_index[meet]:
                    for ans in kg_index[meet][lr]:
                        if not _try_add(hop, pref + [(meet, lr, str(ans))]):
                            break
                if total_added >= max_reasoning_paths:
                    break

    hop1 = paths_by_hop.get(1, [])
    hop2 = paths_by_hop.get(2, [])
    hop3 = paths_by_hop.get(3, [])
    hop4 = paths_by_hop.get(4, [])

    detail["hop_counts"] = {
        1: len(hop1),
        2: len(hop2),
        3: len(hop3),
        4: len(hop4),
    }

    selected_paths: List[List[Tuple[str, str, str]]] = []
    if hop1 or hop2:
        selected_paths = (hop1 + hop2)[:max_reasoning_paths]
        detail["selected_stage"] = "hop1_or_hop2"
    elif hop3:
        selected_paths = hop3[:max_reasoning_paths]
        detail["selected_stage"] = "hop3"
    elif hop4:
        selected_paths = hop4[:max_reasoning_paths]
        detail["selected_stage"] = "hop4"

    detail["returned_count"] = len(selected_paths)
    return (selected_paths, detail) if return_detail else selected_paths

def _dedup_triple_paths(paths: List[List[Tuple[str, str, str]]]) -> List[List[Tuple[str, str, str]]]:
    seen = set()
    out = []
    for p in paths:
        s = _triple_path_to_str(p)
        if s and s not in seen:
            seen.add(s)
            out.append(p)
    return out


def retrieve_centered_reasoning_paths_fallback(
    q_entities: List[str],
    kg_index: Dict[str, Dict[str, Set[str]]],
    *,
    max_center_hops: int = 4,
    max_reasoning_paths: int = 5000,
    return_detail: bool = False,
):
    detail = {
        "prefix_depth_counts": {d: 0 for d in range(1, max_center_hops + 1)},
        "hop_counts": {h: 0 for h in range(1, 5)},
        "returned_count": 0,
        "selected_stage": "none",
    }

    if not q_entities:
        return ([], detail) if return_detail else []

    max_center_hops = max(1, min(int(max_center_hops), 4))

    prefix_buckets = _enumerate_prefix_paths(
        start_entities=q_entities,
        kg_index=kg_index,
        max_depth=max_center_hops,
        forbid_repeat_node=True,
    )
    detail["prefix_depth_counts"] = _count_bucket_sizes(prefix_buckets, max_center_hops)

    hop1 = _dedup_triple_paths(prefix_buckets.get(1, []))
    hop2 = _dedup_triple_paths(prefix_buckets.get(2, []))
    hop3 = _dedup_triple_paths(prefix_buckets.get(3, []))
    hop4 = _dedup_triple_paths(prefix_buckets.get(4, []))

    detail["hop_counts"] = {
        1: len(hop1),
        2: len(hop2),
        3: len(hop3),
        4: len(hop4),
    }

    selected_paths: List[List[Tuple[str, str, str]]] = []
    if hop1 or hop2:
        selected_paths = (hop1 + hop2)[:max_reasoning_paths]
        detail["selected_stage"] = "hop1_or_hop2"
    elif hop3:
        selected_paths = hop3[:max_reasoning_paths]
        detail["selected_stage"] = "hop3"
    elif hop4:
        selected_paths = hop4[:max_reasoning_paths]
        detail["selected_stage"] = "hop4"

    detail["returned_count"] = len(selected_paths)
    return (selected_paths, detail) if return_detail else selected_paths

# =========================
# Inference: rank TYPE PATH -> (relseq) -> search_paths -> reasoning rules
# =========================
@torch.no_grad()
def infer_reasoning_rules_and_eval(
    ds_split,
    rank_sbert,
    output_path: str,
    args=None,
    *,
    tailtype_map: Dict[str, Set[str]],
    tailtype2lastrels: Dict[str, Set[str]],
    global_kg_index: Dict[str, Set[str]],
    global_graph_triples: Dict[str, Set[str]],
    global_rel2heads: Dict[str, Set[str]],
):
    ensure_parent_dir(output_path)

    base_dir = os.path.dirname(output_path)
    all_results_path = os.path.join(base_dir, "all_results_bidirectional.jsonl")
    predictions_eval_results_path = os.path.join(base_dir, "predictions_eval_results_bidirectional.json")
    wrong_results_path = os.path.join(base_dir, "wrong_results_bidirectional.json")
    wrong_results_analyze_path = os.path.join(base_dir, "wrong_results_analyze_bidirectional.json")
    path_stage_stats_json_path = os.path.join(base_dir, "path_stage_stats_bidirectional.json")
    path_stage_stats_txt_path = os.path.join(base_dir, "path_stage_stats_bidirectional.txt")

    stage_stats: Dict[str, List[int]] = defaultdict(list)
    fallback_enter_count = 0

    with open(output_path, "w", encoding="utf-8") as fout_jsonl, \
         open(all_results_path, "w", encoding="utf-8") as fout_jsonl2:

        first = True
        n_written = 0

        for sample in tqdm(ds_split, desc="[infer] bidirectional retrieval (no LLM)"):
            qid = str(sample.get("id", "") or "")
            question = str(sample.get("question", "") or "")
            q_entity = [str(e) for e in (sample.get("q_entity", []) or [])]
            answer = sample.get("answer", "")

            graph_triples = sample.get("graph", []) or []
            kg_index_used = global_kg_index

            tail_types = tailtype_map.get(qid, set()) or set()

            reasoning_paths_str: List[str] = []
            reasoning_path_scores: List[float] = []
            candidate_answers: List[str] = []
            triple_paths: List[List[Tuple[str, str, str]]] = []
            triple_path_scores: List[float] = []
            path_source = "none"

            bidirectional_detail = {
                "tail_type_count": len(tail_types),
                "candidate_last_rel_count": 0,
                "valid_last_rel_count": 0,
                "prefix_depth_counts": {d: 0 for d in range(1, args.max_hops)},
                "hop_counts": {h: 0 for h in range(1, 5)},
                "returned_count": 0,
                "selected_stage": "none",
            }
            fallback_detail = {
                "prefix_depth_counts": {d: 0 for d in range(1, args.max_hops + 1)},
                "hop_counts": {h: 0 for h in range(1, 5)},
                "returned_count": 0,
                "selected_stage": "none",
            }
            pre_rerank_count = 0

            if qid and question.strip() and q_entity:
                if tail_types:
                    triple_paths, bidirectional_detail = bidirectional_retrieve_reasoning_paths(
                        q_entities=q_entity,
                        tail_types=tail_types,
                        tailtype2lastrels=tailtype2lastrels,
                        kg_index=kg_index_used,
                        global_rel2heads=global_rel2heads,
                        max_forward_depth=args.max_hops - 1,
                        keep_max_hops=args.max_hops,
                        return_detail=True,
                    )
                    if triple_paths:
                        path_source = "bidirectional"

                if not triple_paths:
                    fallback_enter_count += 1
                    triple_paths, fallback_detail = retrieve_centered_reasoning_paths_fallback(
                        q_entities=q_entity,
                        kg_index=kg_index_used,
                        max_center_hops=args.max_hops,
                        max_reasoning_paths=args.max_reasoning_paths,
                        return_detail=True,
                    )
                    if triple_paths:
                        path_source = "center_fallback"

                pre_rerank_count = len(triple_paths)

                if triple_paths:
                    triple_paths, triple_path_scores = _rerank_triple_paths_by_similarity(
                        question=question,
                        triple_paths=triple_paths,
                        rank_sbert=rank_sbert,
                        batch_size=args.path_rank_batch_size,
                    )
                    triple_paths = triple_paths[:args.max_reasoning_paths]
                    triple_path_scores = triple_path_scores[:args.max_reasoning_paths]

                seen_path = set()
                for tp, score in zip(triple_paths, triple_path_scores):
                    s = _triple_path_to_str(tp)
                    if s and s not in seen_path:
                        seen_path.add(s)
                        reasoning_paths_str.append(s)
                        reasoning_path_scores.append(float(score))

                seen_tail = set()
                for s in reasoning_paths_str:
                    parts = tokenize_arrow(s)
                    if parts:
                        tail = str(parts[-1]).strip()
                        if tail and tail not in seen_tail:
                            seen_tail.add(tail)
                            candidate_answers.append(tail)

            _update_stage_stats(stage_stats, "01_bidirectional_tail_type_count", bidirectional_detail["tail_type_count"])
            _update_stage_stats(stage_stats, "02_bidirectional_candidate_last_rel_count", bidirectional_detail["candidate_last_rel_count"])
            _update_stage_stats(stage_stats, "03_bidirectional_valid_last_rel_count", bidirectional_detail["valid_last_rel_count"])

            for d in range(1, args.max_hops):
                _update_stage_stats(
                    stage_stats,
                    f"04_bidirectional_prefix_depth_{d}_count",
                    bidirectional_detail["prefix_depth_counts"].get(d, 0),
                )

            for h in range(1, 5):
                _update_stage_stats(
                    stage_stats,
                    f"05_bidirectional_hop_{h}_count",
                    bidirectional_detail["hop_counts"].get(h, 0),
                )

            _update_stage_stats(stage_stats, "06_bidirectional_returned_count", bidirectional_detail["returned_count"])

            for d in range(1, args.max_hops + 1):
                _update_stage_stats(
                    stage_stats,
                    f"07_fallback_prefix_depth_{d}_count",
                    fallback_detail["prefix_depth_counts"].get(d, 0),
                )

            for h in range(1, 5):
                _update_stage_stats(
                    stage_stats,
                    f"08_fallback_hop_{h}_count",
                    fallback_detail["hop_counts"].get(h, 0),
                )

            _update_stage_stats(stage_stats, "09_fallback_returned_count", fallback_detail["returned_count"])
            _update_stage_stats(stage_stats, "10_pre_rerank_triple_paths_count", pre_rerank_count)
            _update_stage_stats(stage_stats, "11_post_rerank_triple_paths_count", len(triple_paths))
            _update_stage_stats(stage_stats, "12_final_reasoning_paths_count", len(reasoning_paths_str))
            _update_stage_stats(stage_stats, "13_final_candidate_answers_count", len(candidate_answers))

            item = {
                "id": qid,
                "question": question,
                "q_entity": q_entity,
                "answer": answer,
                "pred_tail_types": sorted(list(tail_types)) if tail_types else [],
                "last_hop_rels": sorted(list({r for tt in tail_types for r in (tailtype2lastrels.get(tt, set()) or set())})) if tail_types else [],
                "path_source": path_source,
                "reasoning_paths_str": reasoning_paths_str,
                "reasoning_path_scores": reasoning_path_scores,
                "prediction": candidate_answers,
                "path_stage_counts": {
                    "bidirectional": bidirectional_detail,
                    "fallback": fallback_detail,
                    "pre_rerank_triple_paths_count": pre_rerank_count,
                    "post_rerank_triple_paths_count": len(triple_paths),
                    "final_reasoning_paths_count": len(reasoning_paths_str),
                    "final_candidate_answers_count": len(candidate_answers),
                },
                "entered_center_fallback": (path_source == "center_fallback") or (fallback_detail["returned_count"] >= 0 and not bidirectional_detail["returned_count"]),
            }

            fout_jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
            if not first:
                fout_jsonl2.write("\n")
            first = False
            fout_jsonl2.write(json.dumps(item, ensure_ascii=False))
            n_written += 1

    print(f"Wrote {n_written} records to {output_path}")
    print(f"Also wrote JSON array to {all_results_path}")

    _dump_stage_stats(stage_stats, path_stage_stats_json_path, path_stage_stats_txt_path)
    print(f"Wrote path-stage statistics to {path_stage_stats_json_path}")
    print(f"Wrote path-stage statistics text to {path_stage_stats_txt_path}")
    print(f"{fallback_enter_count} questions entered retrieve_centered_reasoning_paths_fallback")

    eval_answer_result(all_results_path, predictions_eval_results_path, wrong_results_path)

    wrong_items = []
    if os.path.exists(wrong_results_path):
        with open(wrong_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    wrong_items.append(json.loads(line))

    wrong_ids = set(item.get("id", "") for item in wrong_items if isinstance(item, dict))
    with open(wrong_results_analyze_path, "w", encoding="utf-8") as f:
        for s in ds_split:
            if str(s.get("id", "") or "") in wrong_ids:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Extracted {len(wrong_ids)} wrong samples to {wrong_results_analyze_path}")


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_path", type=str, default="datasets/RoG-webqsp")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--test_split", type=str, default="test")
    ap.add_argument(
        "--tailtype_path_test",
        type=str,
        default="outputs/tail_type_predictions/RoG-webqsp/data/tail_type_model_webqsp/test/predictions_1_False.jsonl",
    )
    ap.add_argument("--output_path", type=str,
                    default="outputs/retrieval/RoG-webqsp/predictions_from_filtered_type_paths.jsonl")
    ap.add_argument(
        "--ontology_triples_path",
        type=str,
        default="datasets/ontology_triples_general_buquan_final.json",
        help="JSON file of ontology triples: [[head_type, relation, tail_type], ...]",
    )
    ap.add_argument("--max_reasoning_paths", type=int, default=256)
    ap.add_argument("--enable_path_rank_sbert", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--path_rank_model", type=str, default="all-mpnet-base-v2")
    ap.add_argument("--path_rank_model_path", type=str, default="models/all-mpnet-base-v2")
    ap.add_argument("--path_rank_batch_size", type=int, default=256)
    ap.add_argument("--max_hops", type=int, default=2)

    args = ap.parse_args()

    # ===== infer =====
    ds_test = load_dataset(args.dataset_path, split=args.test_split)

    # 1) load tail types
    tailtype_map_test = load_tailtypes_jsonl(args.tailtype_path_test, include_ground_tail_types=False)

    # 2) load ontology mapping: tail_type -> last_rel
    onto_triples = load_ontology_triples(args.ontology_triples_path)
    tailtype2lastrels = build_tailtype2lastrels(onto_triples)

    global_kg_index = defaultdict(lambda: defaultdict(set))
    global_graph_triples = []

    for sample in ds_test:
        for h, r, t in sample.get("graph", []) or []:
            h, r, t = str(h), str(r), str(t)
            global_kg_index[h][str(r)].add(t)
            global_graph_triples.append((h, str(r), t))

    # Precompute relation -> heads index once.
    # Only relations that may serve as answer-side final-hop relations are needed.
    all_candidate_last_rels: Set[str] = set()
    for rels in tailtype2lastrels.values():
        for r in rels or set():
            r = str(r).strip()
            if r:
                all_candidate_last_rels.add(r)

    global_rel2heads = build_rel2heads_for_rels(
        global_graph_triples,
        all_candidate_last_rels,
    )

    print(
        f"[precompute] global_rel2heads built: "
        f"{len(global_rel2heads)} relations, "
        f"{sum(len(v) for v in global_rel2heads.values())} relation-head entries",
        flush=True,
    )

    # 3) Load Sentence-BERT for path ranking.
    rank_sbert = None
    if bool(getattr(args, "enable_path_rank_sbert", True)):
        model_name_or_path = (args.path_rank_model_path or "").strip() or args.path_rank_model
        try:
            rank_sbert = SentenceTransformer(model_name_or_path)
            print(f"[path-rank-sbert] loaded: {model_name_or_path}", flush=True)
        except Exception as e:
            rank_sbert = None
            print(f"[WARN] path rank sbert load failed, disable similarity ranking. err={repr(e)}", flush=True)

    infer_reasoning_rules_and_eval(
        ds_split=ds_test,
        rank_sbert=rank_sbert,
        output_path=args.output_path,
        args=args,
        tailtype_map=tailtype_map_test,
        tailtype2lastrels=tailtype2lastrels,
        global_kg_index=global_kg_index,
        global_graph_triples=global_graph_triples,
        global_rel2heads=global_rel2heads,
    )

if __name__ == "__main__":
    main()

