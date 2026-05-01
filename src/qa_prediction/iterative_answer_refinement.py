
import sys
import os
import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
from functools import partial
import time

from tqdm import tqdm
from datasets import load_dataset
from utils.qa_utils import eval_answer_result
import utils
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import io
from contextlib import redirect_stdout


# ========== OpenAI ==========
# Requires OpenAI Python SDK v1: pip install --upgrade openai
from openai import OpenAI, BadRequestError
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
SEP_PATTERN = re.compile(r"[\n,;\uFF0C\u3001]+")

PATH_REFINE_PROMPT = r"""
You are SelfReflect in an agent loop for KG-QA.

Given:
- Question
- Topic entities
- Proposed answers (from solver)
- Evidence (indexed reasoning paths subset)
- Candidate answers subset (may be noisy / incomplete)
- Allowed tail entity types (optional)

Your job:
1) Decide confidence of current answers: conf ∈ {{1,2,3}}
   - 3: the current top answer is likely correct and there is no clear contradiction
   - 2: there is a limited but concrete issue; some answers may still be plausible
   - 1: the answers are empty, clearly wrong, or unsupported

2) If not conf=3, tell what is missing using ONE of the following tags:
   - "solver_empty"
   - "top1_conflict_need_evidence"
   - "answer_set_noisy"
   - "done"   # use only if conf=3

3) Return explicit action fields for the next round:
   - keep_answers: subset of proposed answers to keep
   - forbid_answers: subset of proposed answers to avoid next round
   - prioritize_path_ids: indexed paths that should be focused on next round
   - supplement_path_ids: indexed paths that should also be shown next round
   - drop_path_ids: indexed paths that are misleading / weak / should be avoided next round
   - decision_summary: one short operational sentence for the next solver call

STRICT OUTPUT: return ONLY valid JSON object:
{{
  "conf": 1,
  "missing": "solver_empty",
  "keep_answers": [],
  "forbid_answers": [],
  "prioritize_path_ids": [],
  "supplement_path_ids": [],
  "drop_path_ids": [],
  "decision_summary": "Regenerate with better supported answers."
}}

Rules:
- Do NOT output Freebase IDs like m.xxx / g.xxx.
- keep_answers MUST be a subset of Proposed answers (string match after trimming).
- forbid_answers should be conservative: only include clearly wrong answers.
- Path ids must be integers from the indexed reasoning paths shown below.
- prioritize_path_ids should contain the most important supporting paths for the next round.
- supplement_path_ids should contain extra paths that may help resolve conflicts or uncertainty.
- drop_path_ids should contain only clearly misleading / irrelevant / contradicted paths.
- If the question contains constraint words such as recent, latest, last, first, earliest, newest, or similar, do not judge only by whether the entity relation chain is plausible; also verify that each kept answer satisfies the required temporal or ordinal constraint.
- If the question expects a single answer, return only the single best-supported answer.
- If the question expects multiple answers, keep the strongly supported and plausible answers only.
- Use the reasoning paths and, if needed, relevant world knowledge to choose the most appropriate answer.

Question: {question}
Allowed tail entity types (optional):
{allowed_tail_types}

Proposed answers: {proposed_answer}

Evidence - Indexed reasoning paths:
{reasoning_paths}
"""

ANSWER_REVISION_TAIL = r"""
### REFINE ACTIONS (HIGH PRIORITY)
{refine_feedback}

Rules for this round:
- Keep answers in Keep Answers if they are still supported.
- Do NOT output any answer from Forbidden Answers.
- Focus on the paths listed in Prioritize Paths first.
- Also consider Supplement Paths if they help resolve conflicts.
- Avoid relying on Drop Paths.
- Re-evaluate from scratch; do NOT simply repeat the previous answer.
"""

# ----------------- IO -----------------
def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w", encoding="utf-8")
        return fout, []
    else:
        processed_results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    results = json.loads(line)
                    processed_results.append(results["id"])
                except Exception as e:
                    raise ValueError("Error in line: " + line) from e
        fout = open(path, "a", encoding="utf-8")
        return fout, processed_results

def _path_hops(p: str) -> int:
    """
    Reasoning path examples:
      e0->r1->e1                parts=3  hops=1
      e0->r1->e1->r2->e2        parts=5  hops=2
    """
    if not p:
        return 0
    s = str(p).strip()
    if not s:
        return 0
    parts = [x for x in s.split("->") if x != ""]
    if len(parts) < 3:
        return 0
    return max(0, (len(parts) - 1) // 2)


def _filter_reasoning_paths_by_hops(rp_list, *, trigger_len: int = 10):
    rp = [str(x).strip() for x in (rp_list or []) if str(x).strip()]
    if len(rp) <= trigger_len:
        return rp

    hop12 = [p for p in rp if _path_hops(p) in (1, 2)]
    if hop12:
        return hop12

    hop3 = [p for p in rp if _path_hops(p) == 3]
    if hop3:
        return hop3

    return rp

def rerank_paths_mpnet(
    question: str,
    paths: List[str],
    mpnet_model: SentenceTransformer,
    batch_size: int = 128,
) -> List[str]:
    """
    Use all-mpnet-base-v2 to sort paths in descending order by cosine similarity.
    """
    if not paths:
        return []

    clean_paths = [p for p in paths if isinstance(p, str) and p.strip()]
    if not clean_paths:
        return []

    q_emb = mpnet_model.encode(
        [question],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
    )  # [1, d]

    p_emb = mpnet_model.encode(
        clean_paths,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )  # [n, d]

    sims = np.dot(p_emb, q_emb[0]).astype(np.float32)  # [n]
    order = np.argsort(-sims)  # descending
    return [clean_paths[i] for i in order]


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        question_to_rule[qid] = {
            "pred_type_paths": data.get("pred_type_paths", []),
            "reasoning_paths_str": data.get("reasoning_paths_str", []),
            "candidate_answers": data.get("prediction", []),
            "pred_tail_types": data.get("pred_tail_types", []),
            "last_node_str": data.get("last_node_str", ""),
            "rules_string": data.get("rules_string", ""),
            "reference_template": data.get("ReferenceTemplate", data.get("reference_template", "")),
        }

    def find_rule(sample):
        qid = sample["id"]
        r = question_to_rule.get(qid, {})
        sample.setdefault("predicted_paths", [])
        sample.setdefault("ground_paths", [])
        sample["filtered_type_paths"] = r.get("pred_type_paths", [])

        rp_raw = r.get("reasoning_paths_str", [])
        sample["reasoning_paths_str"] = _filter_reasoning_paths_by_hops(rp_raw, trigger_len=10)

        sample["candidate_answers"] = r.get("candidate_answers", [])
        sample["last_node_str"] = r.get("last_node_str", "")
        sample["rules_string"] = r.get("rules_string", "")
        sample["reference_template"] = r.get("reference_template", "")
        sample["pred_tail_types"] = r.get("pred_tail_types", [])

        return sample

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x.get("reasoning_paths_str", []) or []) > 0 or len(x.get("candidate_answers", []) or []) > 0,
            num_proc=n_proc
        )
    return qa_dataset

JSON_ARR_RE = re.compile(r"\[[\s\S]*\]")
MID_RE = re.compile(r"^(?:/m/|/g/)?[mg]\.[0-9a-zA-Z_]+$")

# =========================
# EvidenceFilter: filter reasoning paths before answering
# =========================

def _select_refine_reasoning_paths_by_answers(
    reasoning_paths_str,
    proposed_answers,
    max_match_paths=30,
    max_extra_paths=10,
):
    rp = [str(x).strip() for x in _as_list(reasoning_paths_str) if str(x).strip()]
    answers = [str(x).strip() for x in _as_list(proposed_answers) if str(x).strip()]

    if not rp:
        return []

    norm_answers = [_nfkc(a) for a in answers if _nfkc(a)]
    matched, seen = [], set()

    for p in rp:
        np = _nfkc(p)
        if any(a in np for a in norm_answers):
            if p not in seen:
                seen.add(p)
                matched.append(p)

    out = matched

    # for p in rp:
    #     if p not in seen:
    #         out.append(p)
    #     if len(out) >= max_match_paths + max_extra_paths:
    #         break

    return out

def _select_followup_reasoning_paths(
    full_reasoning_paths,
    keep_answers=None,
    forbid_answers=None,
    fallback_answers=None,
    missing="",
    max_extra_paths=20,
):
    """
    Select focused paths for the second or third generator round.
    """
    rp = [str(x).strip() for x in _as_list(full_reasoning_paths) if str(x).strip()]
    if not rp:
        return []

    keep_answers = [str(x).strip() for x in _as_list(keep_answers) if str(x).strip()]
    forbid_answers = [str(x).strip() for x in _as_list(forbid_answers) if str(x).strip()]
    fallback_answers = [str(x).strip() for x in _as_list(fallback_answers) if str(x).strip()]

    anchor_answers = keep_answers[:] if keep_answers else fallback_answers[:]

    norm_anchor = [_nfkc(a) for a in anchor_answers if _nfkc(a)]
    norm_forbid = [_nfkc(a) for a in forbid_answers if _nfkc(a)]

    matched_anchor = []
    neutral = []
    forbidden_hit = []

    for p in rp:
        np = _nfkc(p)
        hit_anchor = any(a in np for a in norm_anchor) if norm_anchor else False
        hit_forbid = any(a in np for a in norm_forbid) if norm_forbid else False

        if hit_anchor and not hit_forbid:
            matched_anchor.append(p)
        elif hit_forbid:
            forbidden_hit.append(p)
        else:
            neutral.append(p)

    out = []

    # Prefer paths that match keep or fallback answers.
    if matched_anchor:
        out.extend(matched_anchor)

    if missing == "top1_conflict_need_evidence":
        out.extend(neutral[:max_extra_paths])

    elif missing == "answer_set_noisy":
        out.extend(neutral[: max(6, max_extra_paths // 2)])

    elif not matched_anchor:
        out.extend(neutral[:max_extra_paths])

    else:
        out.extend(neutral[: min(6, max_extra_paths)])

    # Deduplicate
    dedup = []
    seen = set()
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)

    return dedup

def _format_indexed_paths_block(paths: list) -> str:
    lines = []
    for i, p in enumerate(paths, 1):
        lines.append(f"{i}. {p}")
    return "\n".join(lines) if lines else "(empty)"

def _extract_first_json_obj(text: str) -> dict:
    """
    Robustly extract the FIRST JSON object from a text.
    Handles markdown fences and extra chatter.
    Returns {} if failed.
    """
    if not text:
        return {}
    s = text.strip().lstrip("\ufeff")

    # Strip markdown fences if any
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1].strip()
            if s.startswith("json"):
                s = s[4:].strip()

    # Try direct
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Try raw_decode from first '{'
    start = s.find("{")
    if start < 0:
        return {}
    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(s[start:])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _as_str_list(x) -> list:
    return [str(v).strip() for v in _as_list(x) if str(v).strip()]

def _normalize_int_ids(x) -> list:
    if not isinstance(x, (list, tuple)):
        return []
    out = []
    seen = set()
    for v in x:
        try:
            iv = int(v)
        except Exception:
            continue
        if iv >= 1 and iv not in seen:
            seen.add(iv)
            out.append(iv)
    return out

def _map_ids_to_paths(ids: list, paths: list) -> list:
    ids = _normalize_int_ids(ids)
    if not ids or not paths:
        return []
    n = len(paths)
    picked = []
    seen = set()
    for iv in ids:
        if 1 <= iv <= n:
            p = paths[iv - 1]
            if isinstance(p, str) and p.strip() and p not in seen:
                seen.add(p)
                picked.append(p)
    return picked

def _render_refine_actions_block(missing, keep_answers, forbid_answers,
                                prioritize_paths, supplement_paths, drop_paths,
                                decision_summary):
    lines = []
    lines.append(f"Missing Tag: {str(missing or '').strip() or 'top1_conflict_need_evidence'}")
    lines.append("Keep Answers: " + (json.dumps(keep_answers, ensure_ascii=False) if keep_answers else "[]"))
    lines.append("Forbidden Answers: " + (json.dumps(forbid_answers, ensure_ascii=False) if forbid_answers else "[]"))
    lines.append("Prioritize Paths: " + (json.dumps(prioritize_paths, ensure_ascii=False) if prioritize_paths else "[]"))
    lines.append("Supplement Paths: " + (json.dumps(supplement_paths, ensure_ascii=False) if supplement_paths else "[]"))
    lines.append("Drop Paths: " + (json.dumps(drop_paths, ensure_ascii=False) if drop_paths else "[]"))
    if decision_summary:
        lines.append("Decision Summary: " + str(decision_summary).strip())
    return "\n".join(lines).strip()

def _looks_like_mid(s: str) -> bool:
    s = (s or "").strip()
    return bool(MID_RE.match(s))

_BAD_PREFIX = (
    "from the given options",
    "given the options",
    "i cannot",
    "i can't",
    "however",
    "but strictly",
    "if you'd like",
    "let me know",
    "the correct answer is not among",
    "not among the provided options",
    "not listed among the options",
    "i can provide",
)
def _is_garbage_answer(seg: str) -> bool:
    if not seg:
        return True
    s = _nfkc(seg)
    if len(s) <= 1:
        return True
    if any(s.startswith(p) for p in _BAD_PREFIX):
        return True
    if len(seg) > 120:
        return True
    return False

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _nfkc(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("`", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()

def _best_fuzzy_map(pred_item: str, candidates: list) -> str or None:
    """
    Map a generated answer to the original candidate string while preserving its format.
    """
    if not candidates:
        return None
    norm_pred = _nfkc(pred_item)
    if not norm_pred:
        return None

    norm2raw = { _nfkc(c): c for c in candidates }
    if norm_pred in norm2raw:
        return norm2raw[norm_pred]

    # Token Jaccard
    def toks(s): return set(re.findall(r"[a-z0-9]+", _nfkc(s)))
    pset = toks(pred_item)
    if not pset:
        return None

    best, best_j = None, 0.0
    for c in candidates:
        cset = toks(c)
        if not cset:
            continue
        inter = len(pset & cset)
        union = len(pset | cset)
        j = inter / union if union else 0.0
        if j > best_j:
            best, best_j = c, j
    return best if best_j >= 0.6 else None


def _parse_json_array(text: str) -> list:
    if not text:
        return []
    m = JSON_ARR_RE.search(text)
    if not m:
        return []
    frag = m.group(0)
    try:
        data = json.loads(frag)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return []

PROMPT_TEMPLATE = """{instruction}

{input}"""

COT_SUFFIX = " Let's think it step by step."
EACH_LINE_SUFFIX = " Please return each answer in a new line."

QUESTION_TMPL = "Question:\n{question}"
GRAPH_CONTEXT_TMPL = "Reasoning Paths:\n{context}\n\n"
CHOICES_TMPL = "\nChoices:\n{choices}"

def _build_generator_prompt(
    *,
    question: str,
    reasoning_paths: list,
    candidate_answers_from_refine: list,
    allowed_tail_types: list,
    forbidden_answers: list,
    feedback_block: str,
    explain: bool,
    cot: bool,
    each_line: bool,
    followup_mode: bool,
) -> str:
    rp = [str(x).strip() for x in _as_list(reasoning_paths) if str(x).strip()]
    refine_cands = [str(x).strip() for x in _as_list(candidate_answers_from_refine) if str(x).strip()]
    allowed_tail_types = _as_str_list(allowed_tail_types or [])
    forbidden_answers = [str(x).strip() for x in _as_list(forbidden_answers) if str(x).strip()]
    feedback_text = str(feedback_block or "").strip()

    if explain:
        answer_req = "Please answer the given question and explain why."
    else:
        answer_req = "Please answer the given question. Please keep the answer as simple as possible and only return answers."

    if rp and feedback_text and allowed_tail_types:
        prefix = "Based on the provided reasoning paths, feedback and allowed answer types, "
    elif rp and feedback_text:
        prefix = "Based on the provided reasoning paths and feedback, "
    elif rp and allowed_tail_types:
        prefix = "Based on the provided reasoning paths and allowed answer types, "
    elif feedback_text and allowed_tail_types:
        prefix = "Based on the provided feedback and allowed answer types, "
    elif rp:
        prefix = "Based on the provided reasoning paths, "
    elif feedback_text:
        prefix = "Based on the provided feedback, "
    elif allowed_tail_types:
        prefix = "Based on the provided allowed answer types, "
    else:
        prefix = ""

    if feedback_text:
        rule_parts = [
            prefix + answer_req,
            "\n- If feedback is provided, first process it internally using the relevant reasoning paths and your own general knowledge (not external sources), then answer the question directly without showing that process."
        ]
    else:
        rule_parts = [
            prefix + answer_req
        ]

    rule_parts.extend([
        "\n- Do not output uncertainty or refusal statements; always return the best-supported concrete answer.",
        "\n- Do NOT output Freebase IDs such as m.xxx or g.xxx. Output human-readable entity names or literal values only.",
        # "\n- If the question specifies a year/date (e.g., 'in 2011'), answer for that time.",
        # "\n- For dates, prefer ISO format YYYY-MM-DD if applicable.",
        "\n- If multiple answers, output each on a new line.",
    ])

    if refine_cands:
        rule_parts.append(
            "\n- Prefer the candidate answers if they are consistent with the provided paths; otherwise ignore them and answer directly."
        )

    if allowed_tail_types:
        rule_parts.append(
            "\n- The final answer entity/entities MUST be instances of one of the allowed Freebase tail entity types listed later in the prompt. Do NOT output the type strings themselves."
        )

    if forbidden_answers:
        rule_parts.append(
            "\n- Never output any answer that appears in the Forbidden Answers list."
        )

    instruction_text = " ".join(rule_parts)
    if cot:
        instruction_text += COT_SUFFIX
    if each_line:
        instruction_text += EACH_LINE_SUFFIX

    instruction_block = "### INSTRUCTION\n" + instruction_text

    body_parts = [f"Question:\n{question}"]

    if feedback_text:
        body_parts.append("Feedback:\n" + feedback_text)

    if rp:
        body_parts.append("Reasoning Paths:\n" + "\n".join(rp))

    if refine_cands:
        title = "Candidate Answers"
        body_parts.append(title + ":\n" + "\n".join(refine_cands))

    if allowed_tail_types:
        body_parts.append(
            "Allowed Answer types (answers MUST be instances of these types):\n"
            + "\n".join(allowed_tail_types)
        )

    if forbidden_answers:
        body_parts.append("Forbidden Answers:\n" + "\n".join(forbidden_answers))

    return instruction_block + "\n\n" + "\n\n".join(body_parts)

def build_instruction(has_rp: bool, has_choices: bool, explain: bool,
                      cot: bool, each_line: bool, allowed_tail_types: list | None = None, followup_mode: bool = False,) -> str:
    if explain:
        base = "Please answer the given question and explain why."
    else:
        base = "Please answer the given question. Please keep the answer as simple as possible and only return answers."

    if has_rp:
        instruction = "Based on the provided reasoning paths, " + base
    else:
        instruction = base

    if has_choices:
        instruction += (
            " Prefer the candidate answers if they are consistent with the provided paths; "
            "otherwise ignore them and answer directly."
        )

    instruction += (
        " IMPORTANT: Do NOT output Freebase IDs such as m.xxx or g.xxx. "
        "Output human-readable entity names or literal values only. "
        "If the question specifies a year/date (e.g., 'in 2011'), answer for that time. "
        "For dates, prefer ISO format YYYY-MM-DD if applicable. "
        "If multiple answers, output each on a new line."
    )

    allowed_tail_types = _as_str_list(allowed_tail_types or [])

    if allowed_tail_types:
        instruction += (
            "\n### STRICT TYPE CONSTRAINT:\n"
            "The final answer entity/entities MUST be instances of ONE OF the following Freebase tail entity types. "
            "Do NOT output the type strings themselves.\n"
        )

    if cot:
        instruction += COT_SUFFIX
    if each_line:
        instruction += EACH_LINE_SUFFIX

    return instruction


def build_messages(
    question,
    q_entities,
    candidate_answers,
    reasoning_paths_str,
    choices,
    pred_tail_types,
    args,
    revision_tail,
    followup_mode: bool = False,
    preferred_answers=None,
    forbidden_answers_for_prompt=None,
):
    question = str(question or "").strip()
    if question and (not question.endswith("?")):
        question += "?"

    # -------- Candidate Answers from refine--------
    if followup_mode:
        retained_refine_candidates = [str(x).strip() for x in _as_list(preferred_answers) if str(x).strip()]
    else:
        if _as_list(choices):
            retained_refine_candidates = [str(x).strip() for x in _as_list(choices) if str(x).strip()]
        else:
            retained_refine_candidates = [str(x).strip() for x in _as_list(candidate_answers) if str(x).strip()]

    avoid_mid = bool(getattr(args, "avoid_mid_in_prompt", True)) if args is not None else True
    if avoid_mid:
        retained_refine_candidates = [c for c in retained_refine_candidates if not _looks_like_mid(c)]

    seen = set()
    dedup_retained = []
    for c in retained_refine_candidates:
        if c not in seen:
            seen.add(c)
            dedup_retained.append(c)

    max_choices = int(getattr(args, "max_choices_in_prompt", 200)) if args is not None else 200
    if max_choices and len(dedup_retained) > max_choices:
        dedup_retained = dedup_retained[:max_choices]

    rp = [str(x).strip() for x in _as_list(reasoning_paths_str) if str(x).strip()]
    max_rp = int(getattr(args, "max_reasoning_paths_in_prompt", 200)) if args is not None else 200
    if max_rp and len(rp) > max_rp:
        rp = rp[:max_rp]

    explain = bool(getattr(args, "explain", False)) if args is not None else False
    cot = bool(getattr(args, "cot", False)) if args is not None else False
    each_line = bool(getattr(args, "each_line", False)) if args is not None else False
    allowed_types = _as_str_list(pred_tail_types)
    feedback_block = str(revision_tail or "").strip()

    prompt = _build_generator_prompt(
        question=question,
        reasoning_paths=rp,
        candidate_answers_from_refine=dedup_retained,
        allowed_tail_types=allowed_types,
        forbidden_answers=forbidden_answers_for_prompt or [],
        feedback_block=feedback_block,
        explain=explain,
        cot=cot,
        each_line=each_line,
        followup_mode=followup_mode,
    )

    system = (
        "You are a helpful and rigorous QA assistant. Answer the question based on the question and the relevant evidence."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

def _norm_ans(a: str) -> str:
    s = _nfkc(a or "")
    s = re.sub(r"[\"'“”‘’`]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _same_answer_list(a, b) -> bool:
    def _norm_list(x):
        out = []
        seen = set()
        for v in (x or []):
            s = _norm_ans(str(v).strip())
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out
    return _norm_list(a) == _norm_list(b)

def _choose_adaptive_iter(prediction_after_iter, natural_stop_iter, natural_stop_reason, fallback_round=1):
    """
    Adaptively choose the final iteration.
    """
    p1 = prediction_after_iter.get("1", []) or []
    p2 = prediction_after_iter.get("2", []) or []
    p3 = prediction_after_iter.get("3", []) or []

    if natural_stop_reason in {"refine_conf_3", "early_stop_no_change_after_refine"} and natural_stop_iter in {1, 2, 3}:
        return int(natural_stop_iter), natural_stop_reason

    if _same_answer_list(p1, p2):
        return 1, "adaptive_stable_iter1_iter2"

    if _same_answer_list(p2, p3):
        return 2, "adaptive_stable_iter2_iter3"

    if fallback_round not in {1, 2, 3}:
        fallback_round = 1
    return int(fallback_round), f"adaptive_fallback_iter_{fallback_round}"


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_eval_metrics(work_dir, topk=-1):
    result_name = f"eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = os.path.join(work_dir, result_name)

    with open(eval_result_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    def _extract(metric_name):
        m = re.search(rf"{metric_name}:\s*([0-9]+(?:\.[0-9]+)?)", text)
        return float(m.group(1)) if m else None

    result = {
        "accuracy": _extract("Accuracy"),
        "hit": _extract("Hit"),
    }

    f1 = _extract("F1")
    precision = _extract("Precision")
    recall = _extract("Recall")

    if f1 is not None:
        result["f1"] = f1
    if precision is not None:
        result["precision"] = precision
    if recall is not None:
        result["recall"] = recall

    return result

def _format_eval_metrics(result: dict) -> str:
    parts = []
    if result.get("accuracy") is not None:
        parts.append(f"Accuracy: {result['accuracy']:.2f}")
    if result.get("hit") is not None:
        parts.append(f"Hit: {result['hit']:.2f}")
    if result.get("f1") is not None:
        parts.append(f"F1: {result['f1']:.2f}")
    if result.get("precision") is not None:
        parts.append(f"Precision: {result['precision']:.2f}")
    if result.get("recall") is not None:
        parts.append(f"Recall: {result['recall']:.2f}")
    return " ".join(parts)


def _print_eval_panel(title: str, result: dict):
    print(f"[{title}] {_format_eval_metrics(result)}", flush=True)


def _write_eval_metrics_report(summary: dict, output_path: str):
    lines = []

    lines.append("[Final Adaptive Result final_prediction_adaptive]")
    lines.append(_format_eval_metrics(summary.get("final_prediction_adaptive", {})))
    lines.append("")

    lines.append("[Generation Round 1 generate_answers_iter_1]")
    lines.append(_format_eval_metrics(summary.get("generate_answers_iter", {}).get("iter_1", {})))
    lines.append("")
    lines.append("[After Refinement Round 1 prediction_after_iter_1]")
    lines.append(_format_eval_metrics(summary.get("prediction_after_iter", {}).get("iter_1", {})))
    lines.append("")

    lines.append("[Generation Round 2 generate_answers_iter_2]")
    lines.append(_format_eval_metrics(summary.get("generate_answers_iter", {}).get("iter_2", {})))
    lines.append("")
    lines.append("[After Refinement Round 2 prediction_after_iter_2]")
    lines.append(_format_eval_metrics(summary.get("prediction_after_iter", {}).get("iter_2", {})))
    lines.append("")

    lines.append("[Generation Round 3 generate_answers_iter_3]")
    lines.append(_format_eval_metrics(summary.get("generate_answers_iter", {}).get("iter_3", {})))
    lines.append("")
    lines.append("[After Refinement Round 3 prediction_after_iter_3]")
    lines.append(_format_eval_metrics(summary.get("prediction_after_iter", {}).get("iter_3", {})))
    lines.append("")

    lines.append("[adaptive_stats]")
    lines.append(json.dumps(summary.get("adaptive_stats", {}), ensure_ascii=False, indent=2))
    lines.append("")

    lines.append("[wrong_results description]")
    lines.append("wrong_results.json stores the wrong cases of final_prediction_adaptive.")
    lines.append("Temporary wrong files for each generate/prediction round are deleted after evaluation.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _cleanup_files(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def _extract_iter_field(obj, prefix, iter_idx):
    """
    Support the current flat field naming style.
    """
    key = f"{prefix}_{iter_idx}"
    return obj.get(key, []) or []


def _write_eval_view_jsonl(src_jsonl, dst_jsonl, pred_getter):
    with open(src_jsonl, "r", encoding="utf-8") as fin, open(dst_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lite = {
                "id": obj["id"],
                "question": obj.get("question", obj.get("input", "")),
                "input": obj.get("input", obj.get("question", "")),
                "ground_truth": obj.get("ground_truth", []),
                "prediction": pred_getter(obj),
            }
            fout.write(json.dumps(lite, ensure_ascii=False) + "\n")


def _eval_one_view(src_jsonl, pred_getter, tag, work_dir, panel_title=None):
    tmp_jsonl = os.path.join(work_dir, f".tmp_{tag}.jsonl")
    tmp_eval = os.path.join(work_dir, f".tmp_{tag}_eval.json")
    tmp_wrong = os.path.join(work_dir, f".tmp_{tag}_wrong.json")

    _write_eval_view_jsonl(src_jsonl, tmp_jsonl, pred_getter)

    with redirect_stdout(io.StringIO()):
        eval_answer_result(tmp_jsonl, tmp_eval, tmp_wrong)

    result = _read_eval_metrics(work_dir)

    if panel_title:
        _print_eval_panel(panel_title, result)

    _cleanup_files(
        tmp_jsonl,
        tmp_eval,
        tmp_wrong,
        os.path.join(work_dir, "eval_result.txt"),
    )
    return result


def _collect_adaptive_stats(all_results_path):
    cnt = {1: 0, 2: 0, 3: 0}
    reason_cnt = {}

    with open(all_results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            it = int(obj.get("adaptive_stop_iter", 1) or 1)
            reason = str(obj.get("adaptive_stop_reason", "") or "").strip()
            if it not in cnt:
                cnt[it] = 0
            cnt[it] += 1
            reason_cnt[reason] = reason_cnt.get(reason, 0) + 1

    return {
        "adaptive_stop_iter_count": cnt,
        "adaptive_stop_reason_count": reason_cnt,
    }


def build_eval_summary(all_results_path, eval_summary_path, wrong_results_path):
    """
    Keep only three final files.
    """
    work_dir = os.path.dirname(eval_summary_path)

    tmp_final_eval = os.path.join(work_dir, ".tmp_final_eval.jsonl")

    with redirect_stdout(io.StringIO()):
        eval_answer_result(all_results_path, tmp_final_eval, wrong_results_path)

    final_eval = _read_eval_metrics(work_dir)
    _print_eval_panel("Final Adaptive Result final_prediction_adaptive", final_eval)

    _cleanup_files(
        tmp_final_eval,
        os.path.join(work_dir, "eval_result.txt"),
    )

    summary = {
        "final_prediction_adaptive": final_eval,
        "prediction_after_iter": {
            "iter_1": _eval_one_view(
                all_results_path,
                lambda obj: _extract_iter_field(obj, "prediction_after_iter", 1),
                "prediction_iter_1",
                work_dir,
                panel_title="After Refinement Round 1 prediction_after_iter_1",
            ),
            "iter_2": _eval_one_view(
                all_results_path,
                lambda obj: _extract_iter_field(obj, "prediction_after_iter", 2),
                "prediction_iter_2",
                work_dir,
                panel_title="After Refinement Round 2 prediction_after_iter_2",
            ),
            "iter_3": _eval_one_view(
                all_results_path,
                lambda obj: _extract_iter_field(obj, "prediction_after_iter", 3),
                "prediction_iter_3",
                work_dir,
                panel_title="After Refinement Round 3 prediction_after_iter_3",
            ),
        },
        "generate_answers_iter": {
            "iter_1": _eval_one_view(
                all_results_path,
                lambda obj: _extract_iter_field(obj, "generate_answers_iter", 1),
                "generate_iter_1",
                work_dir,
                panel_title="Generation Round 1 generate_answers_iter_1",
            ),
            "iter_2": _eval_one_view(
                all_results_path,
                lambda obj: _extract_iter_field(obj, "generate_answers_iter", 2),
                "generate_iter_2",
                work_dir,
                panel_title="Generation Round 2 generate_answers_iter_2",
            ),
            "iter_3": _eval_one_view(
                all_results_path,
                lambda obj: _extract_iter_field(obj, "generate_answers_iter", 3),
                "generate_iter_3",
                work_dir,
                panel_title="Generation Round 3 generate_answers_iter_3",
            ),
        },
        "adaptive_stats": _collect_adaptive_stats(all_results_path),
        "wrong_results_desc": "wrong_results.json stores the wrong cases of final_prediction_adaptive.",
    }

    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    metrics_txt_path = os.path.join(work_dir, "eval_metrics_all.txt")
    _write_eval_metrics_report(summary, metrics_txt_path)

def _filter_forbidden(items: list, forbidden: list) -> list:
    fset = {_norm_ans(x) for x in (forbidden or []) if _norm_ans(x)}
    out = []
    seen = set()
    for it in items or []:
        s = str(it).strip()
        if not s:
            continue
        ns = _norm_ans(s)
        if not ns or ns in fset:
            continue
        if ns in seen:
            continue
        seen.add(ns)
        out.append(s)
    return out

def _init_openai_client(args) -> OpenAI:
    api_key = (args.openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (args.openai_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    if not api_key:
        raise RuntimeError("OpenAI API key is missing. Set --openai_api_key or env OPENAI_API_KEY.")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client

def _safe_json_loads(s: str):
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    j = _extract_first_json_obj(s)
    return j if isinstance(j, dict) else None

def _map_list_to_subset(items: list, universe: list) -> list:
    """
    Map items to a subset of universe using trim, normalization, and fuzzy matching.
    """
    uni = [str(u).strip() for u in (universe or []) if str(u).strip()]
    if not items or not uni:
        return []

    norm2raw = {_nfkc(u): u for u in uni}  # normalize exact match
    out = []
    seen = set()
    for it in items:
        s = str(it).strip()
        if not s:
            continue
        if _looks_like_mid(s) or _is_garbage_answer(s):
            continue

        ns = _nfkc(s)
        picked = None
        if ns in norm2raw:
            picked = norm2raw[ns]
        else:
            picked = _best_fuzzy_map(s, uni)  # token jaccard >=0.6

        if picked:
            nsp = _nfkc(picked)
            if nsp and nsp not in seen:
                seen.add(nsp)
                out.append(picked)
    return out


def _as_json_str_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, tuple):
        return [str(v).strip() for v in list(x) if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []

def _refine_answers_once(
    client: OpenAI,
    model: str,
    question: str,
    q_entities: list,
    proposed: list,
    reasoning_paths_str: list,
    candidate_answers: list,
    pred_tail_types: list,
    args,
):
    max_rp = int(getattr(args, "max_reasoning_paths_in_refine_prompt", 80)) if args is not None else 80
    max_cand = int(getattr(args, "max_candidates_in_refine_prompt", 60)) if args is not None else 60
    max_types = int(getattr(args, "max_tail_types_in_refine_prompt", 60)) if args is not None else 60

    rp = [str(x).strip() for x in _as_list(reasoning_paths_str) if str(x).strip()]
    cands = [str(x).strip() for x in _as_list(candidate_answers) if str(x).strip()]
    types = [str(x).strip() for x in _as_list(pred_tail_types) if str(x).strip()]

    if max_rp and len(rp) > max_rp:
        rp = rp[:max_rp]
    if max_cand and len(cands) > max_cand:
        cands = cands[:max_cand]
    if max_types and len(types) > max_types:
        types = types[:max_types]

    indexed_rp_block = _format_indexed_paths_block(rp)

    prompt = PATH_REFINE_PROMPT.format(
        question=str(question or "").strip(),
        q_entity=[str(x).strip() for x in _as_list(q_entities) if str(x).strip()],
        proposed_answer=json.dumps(
            [str(x).strip() for x in (proposed or []) if str(x).strip()],
            ensure_ascii=False
        ),
        reasoning_paths=(indexed_rp_block if indexed_rp_block else "(empty)"),
        candidate_answers=("\n".join([c for c in cands if not _looks_like_mid(c)]) if cands else "(empty)"),
        allowed_tail_types=("\n".join(types) if types else "(empty)"),
    )

    resp = _chat_with_retries(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You are a conservative answer verifier. Output ONLY strict JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=float(getattr(args, "openai_refine_temperature", 0.0)),
        max_tokens=512,
        n=1,
    )

    if resp is None:
        fallback_json = {
            "conf": 1,
            "missing": "refine_api_failed",
            "keep_answers": [],
            "forbid_answers": [],
            "prioritize_path_ids": [],
            "supplement_path_ids": [],
            "drop_path_ids": [],
            "prioritize_paths": [],
            "supplement_paths": [],
            "drop_paths": [],
            "decision_summary": "refine failed; regenerate a concise answer."
        }
        feedback = _render_refine_actions_block(
            "refine_api_failed", [], [], [], [], [],
            "refine failed; regenerate a concise answer."
        )
        return 1, "refine_api_failed", [], [], feedback, fallback_json

    txt = (resp.choices[0].message.content or "").strip()
    j = _extract_first_json_obj(txt) or {}
    if not isinstance(j, dict):
        j = {}

    conf = j.get("conf", 2)
    try:
        conf = int(conf)
    except Exception:
        conf = 2
    if conf not in (1, 2, 3):
        conf = 2

    missing = str(j.get("missing", "") or "").strip()
    if not missing:
        missing = "done" if conf == 3 else "top1_conflict_need_evidence"

    keep_raw = _as_str_list(j.get("keep_answers", []))
    forbid_raw = _as_str_list(j.get("forbid_answers", []))

    keep = _map_list_to_subset(keep_raw, proposed)
    forbid = _map_list_to_subset(forbid_raw, proposed)

    prioritize_path_ids = _normalize_int_ids(j.get("prioritize_path_ids", []))
    supplement_path_ids = _normalize_int_ids(j.get("supplement_path_ids", []))
    drop_path_ids = _normalize_int_ids(j.get("drop_path_ids", []))

    prioritize_paths = _map_ids_to_paths(prioritize_path_ids, rp)
    supplement_paths = _map_ids_to_paths(supplement_path_ids, rp)
    drop_paths = _map_ids_to_paths(drop_path_ids, rp)

    decision_summary = str(j.get("decision_summary", "") or "").strip()

    feedback = _render_refine_actions_block(
        missing=missing,
        keep_answers=keep,
        forbid_answers=forbid,
        prioritize_paths=prioritize_paths,
        supplement_paths=supplement_paths,
        drop_paths=drop_paths,
        decision_summary=decision_summary,
    )

    if conf == 3 and not keep:
        keep = _map_list_to_subset(proposed[:1], proposed)

    j["keep_answers"] = keep
    j["forbid_answers"] = forbid
    j["prioritize_path_ids"] = prioritize_path_ids
    j["supplement_path_ids"] = supplement_path_ids
    j["drop_path_ids"] = drop_path_ids
    j["prioritize_paths"] = prioritize_paths
    j["supplement_paths"] = supplement_paths
    j["drop_paths"] = drop_paths
    j["decision_summary"] = decision_summary

    return conf, missing, keep, forbid, feedback, j

def _aggregate_segments_with_votes(segs: list, kg_candidate_answers: list):
    if not segs:
        return [], {}

    readable_cands = [c for c in (kg_candidate_answers or []) if
                      isinstance(c, str) and c.strip() and (not _looks_like_mid(c))]
    readable_cands = [c.strip() for c in readable_cands]

    if readable_cands:
        cand_set = set(readable_cands)
        votes = Counter()
        for s in segs:
            mapped = _best_fuzzy_map(s, readable_cands)
            if mapped is not None and mapped in cand_set:
                votes[mapped] += 1

        if votes:
            ranked = [x for x, _ in votes.most_common()]
            ranked = [x for x in ranked if (not _looks_like_mid(x)) and (not _is_garbage_answer(x))]
            vote_cnt = {k: int(v) for k, v in votes.items()}
            return ranked, vote_cnt

    # fallback: consensus bucket
    norm2originals = defaultdict(Counter)
    for s in segs:
        if _is_garbage_answer(s) or _looks_like_mid(s):
            continue
        norm = _nfkc(s)
        if not norm:
            continue
        norm2originals[norm][s] += 1

    scored = []
    vote_cnt = {}
    for norm, bucket in norm2originals.items():
        cnt = sum(bucket.values())
        best_original, _ = bucket.most_common(1)[0]
        scored.append((cnt, best_original))
        vote_cnt[best_original] = int(cnt)

    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [s for _, s in scored]
    ranked = [x for x in ranked if (not _looks_like_mid(x)) and (not _is_garbage_answer(x))]
    return ranked, vote_cnt

def _chat_with_retries(client: OpenAI, **kwargs):
    max_attempts = 6
    base_sleep = 2.0
    content_filter_retry = 2
    content_filter_sleep = 20.0

    last_err = None
    cf_count = 0

    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**kwargs)

        except BadRequestError as e:
            last_err = e
            err_text = str(e).lower()

            if "content_filter" in err_text or "content management policy" in err_text:
                cf_count += 1
                if cf_count <= content_filter_retry:
                    print(
                        f"[WARN] content_filter hit (retry {cf_count}/{content_filter_retry}), "
                        f"sleep {content_filter_sleep:.1f}s ...",
                        flush=True,
                    )
                    time.sleep(content_filter_sleep)
                    continue

                print(f"[WARN] content_filter persists, skip this request: {e}", flush=True)
                return None

            print(f"[WARN] BadRequestError, skip this request: {e}", flush=True)
            return None

        except RateLimitError as e:
            last_err = e
            sleep_t = base_sleep * attempt
            print(f"[WARN] RateLimitError (attempt {attempt}/{max_attempts}), sleep {sleep_t:.1f}s ...", flush=True)
            time.sleep(sleep_t)
            continue

        except (APITimeoutError, APIConnectionError) as e:
            last_err = e
            sleep_t = base_sleep * attempt
            print(f"[WARN] Temporary network error (attempt {attempt}/{max_attempts}), sleep {sleep_t:.1f}s ...", flush=True)
            time.sleep(sleep_t)
            continue

        except APIError as e:
            last_err = e
            status = getattr(e, "status", None)
            if status is not None and 500 <= int(status) < 600 and attempt < max_attempts:
                sleep_t = base_sleep * attempt
                print(f"[WARN] Server error {status} (attempt {attempt}/{max_attempts}), sleep {sleep_t:.1f}s ...", flush=True)
                time.sleep(sleep_t)
                continue

            print(f"[WARN] APIError, skip this request: {e}", flush=True)
            return None

    print(f"[ERROR] Giving up after retries: {last_err}", flush=True)
    return None


def openai_generate_answers(data, args, mpnet_model=None):
    question = data["question"]

    kg_candidate_answers = [str(x).strip() for x in _as_list(data.get("candidate_answers", [])) if str(x).strip()]

    reasoning_paths_str = _as_list(data.get("reasoning_paths_str", []))
    pred_tail_types = _as_list(data.get("pred_tail_types", []))

    q_entity_raw = data.get("q_entity", [])
    q_entities = [str(x).strip() for x in _as_list(q_entity_raw) if str(x).strip()]

    client = _init_openai_client(args)
    iter_limit = max(1, int(getattr(args, "iter_limit", 3)))
    refine_model = (getattr(args, "openai_refine_model", "") or "").strip() or args.openai_model

    # # mpnet rerank reasoning paths
    if mpnet_model is not None and reasoning_paths_str:
        try:
            bs = int(getattr(args, "mpnet_batch_size", 128))
            reasoning_paths_str = rerank_paths_mpnet(
                question=str(question or ""),
                paths=[str(x) for x in reasoning_paths_str],
                mpnet_model=mpnet_model,
                batch_size=bs,
            )
        except Exception as e:
            print(f"[WARN] mpnet rerank failed, keep original paths. err={repr(e)}", flush=True)

    forbidden_answers = []
    last_refine_feedback = ""
    last_refine_json = {}

    generate_answers_iter = {"1": [], "2": [], "3": []}
    refine_answers_iter = {"1": [], "2": [], "3": []}
    prediction_after_iter = {"1": [], "2": [], "3": []}
    prompt_reasoning_paths_iter = {"1": [], "2": [], "3": []}

    natural_stop_iter = None
    natural_stop_reason = None

    prev_after_refine = None

    # --- Run one generation call. ---
    def _one_call(
            force_no_choices: bool = False,
            revision_tail: str = "",
            iter_reasoning_paths_str=None,
            iter_keep_answers=None,
            followup_mode: bool = False,
    ):

        if followup_mode:
            cand_for_prompt = [str(x).strip() for x in _as_list(iter_keep_answers) if str(x).strip()]
        else:
            cand_for_prompt = _filter_forbidden(kg_candidate_answers, forbidden_answers)

        use_rp = reasoning_paths_str if iter_reasoning_paths_str is None else iter_reasoning_paths_str

        msgs = build_messages(
            question=question,
            q_entities=q_entities,
            candidate_answers=([] if force_no_choices else cand_for_prompt),
            reasoning_paths_str=use_rp,
            choices=([] if (force_no_choices or followup_mode) else data.get("choices", [])),
            pred_tail_types=pred_tail_types,
            args=args,
            revision_tail=revision_tail,
            followup_mode=followup_mode,
            preferred_answers=cand_for_prompt,
            forbidden_answers_for_prompt=forbidden_answers,
        )

        resp = _chat_with_retries(
            client,
            model=args.openai_model,
            messages=msgs,
            temperature=args.openai_temperature,
            max_tokens=args.openai_max_tokens,
            n=max(1, int(args.openai_samples)),
        )

        if resp is None:
            return []

        all_segments = []
        for ch in resp.choices:
            txt = (ch.message.content or "").strip()
            if not txt:
                continue
            pieces = [s.strip(" -*•\t") for s in SEP_PATTERN.split(txt) if s.strip()]
            for p in pieces:
                if not _is_garbage_answer(p):
                    all_segments.append(p)
        return all_segments

    last_completed_key = None
    # ======= Iteration: solver -> refine -> revise =======
    for it in range(iter_limit):
        iter_key = str(it + 1)
        last_completed_key = iter_key
        revision_tail = ""

        # Read fields from the previous refinement JSON; use defaults if missing.
        j = last_refine_json if isinstance(last_refine_json, dict) else {}

        jf = str(last_refine_feedback or "").strip()

        keep_list_prev = [str(x).strip() for x in _as_list(j.get("keep_answers", [])) if str(x).strip()]

        # Merge historical forbidden answers with current refinement forbidden answers.
        fb_list = []
        fb_list.extend([str(x).strip() for x in _as_list(forbidden_answers) if str(x).strip()])
        fb_list.extend([str(x).strip() for x in _as_list(j.get("forbid_answers", [])) if str(x).strip()])

        # Deduplicate
        seen = set()
        fb_dedup = []
        for x in fb_list:
            nx = _norm_ans(x)
            if not nx or nx in seen:
                continue
            seen.add(nx)
            fb_dedup.append(x)

        # Inject constraints when any refinement signal exists.
        if fb_dedup or jf or keep_list_prev:
            revision_tail = ANSWER_REVISION_TAIL.format(
                refine_feedback=jf or _render_refine_actions_block(
                    missing="top1_conflict_need_evidence",
                    keep_answers=keep_list_prev,
                    forbid_answers=fb_dedup,
                    prioritize_paths=[],
                    supplement_paths=[],
                    drop_paths=[],
                    decision_summary="previous answer unstable; refine using evidence",
                ),
            )

        # =====From the second round onward, use focused paths guided by the previous refinement. =====
        iter_reasoning_paths = reasoning_paths_str
        if it >= 1:
            prev_iter_key = str(it)
            prev_generated = generate_answers_iter.get(prev_iter_key, []) or []
            prev_missing = str(j.get("missing", "") or "").strip()

            iter_reasoning_paths = _select_followup_reasoning_paths(
                full_reasoning_paths=reasoning_paths_str,
                keep_answers=keep_list_prev,
                forbid_answers=fb_dedup,
                fallback_answers=prev_generated,
                missing=prev_missing,
                max_extra_paths=int(getattr(args, "max_followup_extra_paths", 20)),
            )

            prioritize_paths_prev = [str(x).strip() for x in _as_list(j.get("prioritize_paths", [])) if str(x).strip()]
            supplement_paths_prev = [str(x).strip() for x in _as_list(j.get("supplement_paths", [])) if str(x).strip()]
            drop_paths_prev = {_nfkc(str(x).strip()) for x in _as_list(j.get("drop_paths", [])) if str(x).strip()}

            merged_paths = []

            for p in prioritize_paths_prev:
                if p not in merged_paths and _nfkc(p) not in drop_paths_prev:
                    merged_paths.append(p)

            for p in iter_reasoning_paths:
                p0 = str(p).strip()
                if p0 and p0 not in merged_paths and _nfkc(p0) not in drop_paths_prev:
                    merged_paths.append(p0)

            for p in supplement_paths_prev:
                if p not in merged_paths and _nfkc(p) not in drop_paths_prev:
                    merged_paths.append(p)

            if merged_paths:
                iter_reasoning_paths = merged_paths

        prompt_reasoning_paths_iter[iter_key] = [str(x) for x in _as_list(iter_reasoning_paths)]

        followup_mode = (it >= 1)

        segs = _one_call(
            force_no_choices=False,
            revision_tail=revision_tail,
            iter_reasoning_paths_str=iter_reasoning_paths,
            iter_keep_answers=keep_list_prev if followup_mode else [],
            followup_mode=followup_mode,
        )

        # segs = _one_call(force_no_choices=False, revision_tail=revision_tail)

        ranked_pred, vote_cnt = _aggregate_segments_with_votes(segs, kg_candidate_answers)
        pred = ranked_pred if ranked_pred else []

        generate_answers_iter[iter_key] = pred[:]

        if not pred:
            refine_answers_iter[iter_key] = []
            prediction_after_iter[iter_key] = []
            last_refine_feedback = _render_refine_actions_block(
                missing="solver_empty",
                keep_answers=[],
                forbid_answers=[],
                prioritize_paths=[],
                supplement_paths=[],
                drop_paths=[],
                decision_summary="Solver returned empty answers; MUST output at least one concrete answer.",
            )
            continue

        refine_reasoning_paths = _select_refine_reasoning_paths_by_answers(
            reasoning_paths_str=reasoning_paths_str,
            proposed_answers=pred,
            max_match_paths=30,
            max_extra_paths=10,
        )

        conf, missing, keep_list, forbid_list, feedback, j_json = _refine_answers_once(
            client=client,
            model=refine_model,
            question=question,
            q_entities=q_entities,
            proposed=pred,
            reasoning_paths_str=refine_reasoning_paths,
            candidate_answers=kg_candidate_answers,
            pred_tail_types=pred_tail_types,
            args=args,
        )
        last_refine_json = j_json

        refine_answers = keep_list[:] if keep_list else (pred[:] if conf == 3 else [])
        refine_answers_iter[iter_key] = refine_answers

        prediction_after_iter[iter_key] = refine_answers[:] if refine_answers else pred[:]

        current_after_refine = prediction_after_iter[iter_key][:]

        # Stop if two consecutive refined outputs are unchanged.
        if prev_after_refine is not None and _same_answer_list(current_after_refine, prev_after_refine):
            natural_stop_iter = it + 1
            natural_stop_reason = "early_stop_no_change_after_refine"
            for nxt in range(it + 2, 4):
                nxt_key = str(nxt)
                generate_answers_iter[nxt_key] = generate_answers_iter[iter_key][:]
                refine_answers_iter[nxt_key] = refine_answers_iter[iter_key][:]
                prediction_after_iter[nxt_key] = prediction_after_iter[iter_key][:]
            break

        prev_after_refine = current_after_refine[:]

        # Stop on conf == 3; otherwise continue to the next generation round.
        if conf == 3:
            natural_stop_iter = it + 1
            natural_stop_reason = "refine_conf_3"

            for nxt in range(it + 2, 4):
                nxt_key = str(nxt)
                generate_answers_iter[nxt_key] = generate_answers_iter[iter_key][:]
                refine_answers_iter[nxt_key] = refine_answers_iter[iter_key][:]
                prediction_after_iter[nxt_key] = prediction_after_iter[iter_key][:]
            break

        # For conf != 3, update forbidden answers and feedback before the next round.
        for a in (forbid_list or []):
            a0 = str(a).strip()
            if a0 and _norm_ans(a0) not in {_norm_ans(x) for x in forbidden_answers}:
                forbidden_answers.append(a0)

        decision_summary = str(j_json.get("decision_summary", "") or "").strip()
        keep_now = [str(x).strip() for x in _as_list(j_json.get("keep_answers", [])) if str(x).strip()]
        forbid_now = [str(x).strip() for x in _as_list(j_json.get("forbid_answers", [])) if str(x).strip()]
        prioritize_now = [str(x).strip() for x in _as_list(j_json.get("prioritize_paths", [])) if str(x).strip()]
        supplement_now = [str(x).strip() for x in _as_list(j_json.get("supplement_paths", [])) if str(x).strip()]
        drop_now = [str(x).strip() for x in _as_list(j_json.get("drop_paths", [])) if str(x).strip()]

        last_refine_feedback = _render_refine_actions_block(
            missing=missing,
            keep_answers=keep_now,
            forbid_answers=forbid_now,
            prioritize_paths=prioritize_now,
            supplement_paths=supplement_now,
            drop_paths=drop_now,
            decision_summary=(decision_summary or "regenerate with stricter consistency constraints"),
        )

    if natural_stop_iter is None:
        if last_completed_key is None:
            last_completed_key = "1"
        natural_stop_iter = int(last_completed_key)
        natural_stop_reason = "reach_iter_limit"

        for nxt in range(natural_stop_iter + 1, 4):
            nxt_key = str(nxt)
            generate_answers_iter[nxt_key] = generate_answers_iter[last_completed_key][:]
            refine_answers_iter[nxt_key] = refine_answers_iter[last_completed_key][:]
            prediction_after_iter[nxt_key] = prediction_after_iter[last_completed_key][:]

    adaptive_iter, adaptive_reason = _choose_adaptive_iter(
        prediction_after_iter=prediction_after_iter,
        natural_stop_iter=natural_stop_iter,
        natural_stop_reason=natural_stop_reason,
        fallback_round=int(getattr(args, "adaptive_fallback_round", 1)),
    )

    final_out = prediction_after_iter[str(adaptive_iter)][:]

    data["iter_debug"] = {
        "forbidden_answers": forbidden_answers[:20],
        "last_refine_feedback": last_refine_feedback,
        "last_refine_json": last_refine_json,
        "generate_answers_iter": generate_answers_iter,
        "refine_answers_iter": refine_answers_iter,
        "prediction_after_iter": prediction_after_iter,
        "natural_stop_iter": natural_stop_iter,
        "natural_stop_reason": natural_stop_reason,
        "adaptive_stop_iter": adaptive_iter,
        "adaptive_stop_reason": adaptive_reason,
    }

    return {
        "prediction": final_out,
        "prediction_after_iter_1": prediction_after_iter["1"],
        "prediction_after_iter_2": prediction_after_iter["2"],
        "prediction_after_iter_3": prediction_after_iter["3"],
        "generate_answers_iter_1": generate_answers_iter["1"],
        "generate_answers_iter_2": generate_answers_iter["2"],
        "generate_answers_iter_3": generate_answers_iter["3"],
        "refine_answers_iter_1": refine_answers_iter["1"],
        "refine_answers_iter_2": refine_answers_iter["2"],
        "refine_answers_iter_3": refine_answers_iter["3"],
        "natural_stop_iter": natural_stop_iter,
        "natural_stop_reason": natural_stop_reason,
        "adaptive_stop_iter": adaptive_iter,
        "adaptive_stop_reason": adaptive_reason,
    }



def prediction(data, processed_list, args, mpnet_model=None):
    q = data["question"]
    gt = data.get("answer", [])
    qid = data["id"]
    if qid in processed_list:
        return None

    try:
        trace = openai_generate_answers(data, args, mpnet_model=mpnet_model)
    except Exception as e:
        print(f"[ERROR] sample failed, skip qid={qid}, err={e}", flush=True)
        trace = {
            "prediction": [],
            "prediction_after_iter_1": [],
            "prediction_after_iter_2": [],
            "prediction_after_iter_3": [],
            "generate_answers_iter_1": [],
            "generate_answers_iter_2": [],
            "generate_answers_iter_3": [],
            "refine_answers_iter_1": [],
            "refine_answers_iter_2": [],
            "refine_answers_iter_3": [],
            "natural_stop_iter": 1,
            "natural_stop_reason": "error",
            "adaptive_stop_iter": 1,
            "adaptive_stop_reason": "error",
        }

    return {
        "id": qid,
        "question": q,
        "prediction": trace["prediction"],
        "prediction_after_iter_1": trace["prediction_after_iter_1"],
        "prediction_after_iter_2": trace["prediction_after_iter_2"],
        "prediction_after_iter_3": trace["prediction_after_iter_3"],
        "generate_answers_iter_1": trace["generate_answers_iter_1"],
        "generate_answers_iter_2": trace["generate_answers_iter_2"],
        "generate_answers_iter_3": trace["generate_answers_iter_3"],
        "refine_answers_iter_1": trace["refine_answers_iter_1"],
        "refine_answers_iter_2": trace["refine_answers_iter_2"],
        "refine_answers_iter_3": trace["refine_answers_iter_3"],
        "natural_stop_iter": trace["natural_stop_iter"],
        "natural_stop_reason": trace["natural_stop_reason"],
        "kg_candidate_answers": data.get("candidate_answers", []),
        "ground_truth": gt,
        "input": q,
        "adaptive_stop_iter": trace["adaptive_stop_iter"],
        "adaptive_stop_reason": trace["adaptive_stop_reason"],
    }


# ----------------- main -----------------
def main(args):
    try:
        probe_client = _init_openai_client(args)
        model_list = probe_client.models.list()
        print("\n[INFO] Available models:", flush=True)
        for m in model_list.data:
            print(" -", m.id, flush=True)
    except Exception as e:
        print(f"[WARN] failed to list models: {e}", flush=True)

    input_file = os.path.join(args.data_path, args.d + "/data")

    dataset = load_dataset(input_file, split=args.split)

    # ===== mpnet model init (only once in main process) =====
    mpnet_model = None
    if bool(getattr(args, "enable_mpnet_rerank", True)):
        try:
            mpnet_model = SentenceTransformer(args.mpnet_path)
            print(f"[mpnet] loaded: {args.mpnet_path}", flush=True)
        except Exception as e:
            mpnet_model = None
            print(f"[WARN] mpnet load failed, disable rerank. err={repr(e)}", flush=True)

    rule_postfix = "openai"
    if args.add_rule:
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        if args.use_true:
            rule_postfix = "ground_rule_openai"
        elif args.use_random:
            rule_postfix = "random_rule_openai"

    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"
    rule_postfix += "qwen2_7B_ds_v3_250324_results_pred"

    print("Load dataset finished.")
    output_dir = os.path.join(args.predict_path, args.d, rule_postfix)
    print("Save results to:", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "args.txt"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    output_file = os.path.join(output_dir, "predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    worker = partial(prediction, processed_list=processed_list, args=args, mpnet_model=mpnet_model)

    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(p.imap(worker, dataset), total=len(dataset)):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res, ensure_ascii=False))
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = worker(data)
            if res is not None:
                if args.debug:
                    print(json.dumps(res, ensure_ascii=False))
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()

    fout.close()

    all_results_path = output_file
    base_dir = os.path.dirname(output_file)

    eval_summary_path = os.path.join(base_dir, "eval_summary.json")
    wrong_results_path = os.path.join(base_dir, "wrong_results.json")

    build_eval_summary(
        all_results_path=all_results_path,
        eval_summary_path=eval_summary_path,
        wrong_results_path=wrong_results_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="datasets")
    parser.add_argument("--ontology_path", type=str, default="datasets/ontology_triples_general_buquan_final.json")
    parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--predict_path", type=str, default="outputs/final_predictions")
    parser.add_argument(
        "--rule_path",
        type=str,
        default="outputs/retrieval/RoG-webqsp/predictions_from_filtered_type_paths.jsonl",
    )

    parser.add_argument("--add_rule", default=True, action="store_true")
    parser.add_argument("--use_true", action="store_true")
    parser.add_argument("--cot", default=False, action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--use_random", action="store_true")
    parser.add_argument("--each_line", action="store_true")
    parser.add_argument("--filter_empty", action="store_true")

    parser.add_argument("--force", "-f", action="store_true", help="force to overwrite the results")
    parser.add_argument("-n", default=1, type=int, help="number of processes")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key. If empty, read from OPENAI_API_KEY.",
    )

    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL.",
    )

    parser.add_argument(
        "--openai_model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-4o"),
        help="OpenAI-compatible model name used for answer refinement.",
    )

    parser.add_argument("--openai_temperature", type=float, default=0.2)
    parser.add_argument("--openai_max_tokens", type=int, default=128)
    parser.add_argument("--openai_samples", type=int, default=1, help="number of samples per question (n for chat.completions)")
    parser.add_argument("--max_return", type=int, default=5, help="max answers to return in prediction array")

    parser.add_argument("--avoid_mid_in_prompt", action="store_true", default=True, help="If set, filter Freebase IDs (m.xxx/g.xxx) out of prompt choices/candidates.")
    parser.add_argument("--max_type_paths_in_prompt", type=int, default=80)
    parser.add_argument("--max_choices_in_prompt", type=int, default=200)
    parser.add_argument("--max_reasoning_paths_in_prompt", type=int, default=256)

    # EvidenceFilter
    parser.add_argument("--enable_evidence_filter", action="store_true", default=True)
    parser.add_argument("--openai_filter_model", type=str, default="", help="If empty, use --openai_model")
    parser.add_argument("--max_paths_in_filter_prompt", type=int, default=200)

    # ===== mpnet rerank =====
    parser.add_argument("--enable_mpnet_rerank", action="store_true", default=True)
    parser.add_argument("--mpnet_path", type=str, default="models/all-mpnet-base-v2")
    parser.add_argument("--mpnet_batch_size", type=int, default=128)

    parser.add_argument("--refine_expand_k", type=int, default=2,
                        help="When refine confirms, append up to K extra high-confidence answers (improve recall/F1).")
    parser.add_argument("--refine_expand_min_votes", type=int, default=2,
                        help="Min vote count for an extra answer to be appended after refine confirmed (protect precision).")

    parser.add_argument("--max_followup_extra_paths", type=int, default=20,
                        help="Extra neutral reasoning paths appended in follow-up rounds.")
    parser.add_argument("--adaptive_fallback_round", type=int, default=1,
                        help="Fallback round used when no natural stop / no stable consecutive iterations.")

    args = parser.parse_args()
    main(args)

