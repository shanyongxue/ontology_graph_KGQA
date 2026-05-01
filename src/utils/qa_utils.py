from collections import OrderedDict
import json
import re
import string
from sklearn.metrics import precision_score
from statistics import mean
import os
from tqdm import tqdm

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_acc(prediction, answer):
    if not answer:
        return 1.0 if not prediction else 0.0

    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    if not answer:
        return 1.0 if not prediction else 0.0

    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def eval_f1(prediction, answer):
    if len(prediction) == 0 or len(answer) == 0:
        return 0, 0, 0
    ans_recalled = 0
    prediction_correct = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            ans_recalled += 1
    recall = ans_recalled / len(answer)
    for p in prediction:
        for a in answer:
            if match(p, a):
                prediction_correct += 1
                break
    precision = prediction_correct / len(prediction)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return (2 * precision * recall) / (precision + recall), precision, recall
def eval_relation_path_result(
    all_results_path: str,
    wrong_results_path: str,
    strict: bool = False,
    topk: int = -1,
    skip_empty: bool = False
):
    data = []
    with open(all_results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    accs, hits, precs, recs, f1s = [], [], [], [], []
    wrong_items, total = [], 0

    for sample in tqdm(data, desc="Evaluating relation paths"):
        total += 1
        sid = sample.get("id")
        question = sample.get("question", "")

        gold_list = [tuple(p) for p in (sample.get("ground_paths") or []) if p]
        pred_list = [tuple(p) for p in (sample.get("prediction")   or []) if p]

        if topk and topk > 0 and len(pred_list) > topk:
            pred_list = pred_list[:topk]

        gold_set, pred_set = set(gold_list), set(pred_list)
        if skip_empty and (not gold_set or not pred_set):
            continue

        inter = gold_set & pred_set
        hit = 1 if inter else 0
        acc = 1 if (pred_set == gold_set) else (1 if hit and not strict else 0)

        prec = (len(inter) / len(pred_set)) if pred_set else 0.0
        rec  = (len(inter) / len(gold_set)) if gold_set else 0.0
        f1   = (2*prec*rec / (prec+rec)) if (prec > 0 and rec > 0) else 0.0

        accs.append(acc); hits.append(hit); precs.append(prec); recs.append(rec); f1s.append(f1)

        is_wrong = (acc == 0) if strict else (hit == 0)
        if is_wrong and gold_set:
            wrong_items.append({
                "id": sid,
                "question": question,
                "ground_paths": [list(p) for p in gold_list],
                "prediction":   [list(p) for p in pred_list],
                "intersection": [list(p) for p in sorted(inter)],
                "missing":      [list(p) for p in sorted(gold_set - pred_set)],
                "extra":        [list(p) for p in sorted(pred_set - gold_set)],
                "acc": acc, "hit": hit, "precision": prec, "recall": rec, "f1": f1,
            })

    os.makedirs(os.path.dirname(wrong_results_path), exist_ok=True)
    with open(wrong_results_path, "w", encoding="utf-8") as f:
        for item in wrong_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    print(
        f"Samples: {total} | "
        f"Accuracy: {avg(accs)*100:.2f}  "
        f"Hit: {avg(hits)*100:.2f}  "
        f"F1: {avg(f1s)*100:.2f}  "
        f"Precision: {avg(precs)*100:.2f}  "
        f"Recall: {avg(recs)*100:.2f}"
    )
    print(f"Wrong cases -> {wrong_results_path} | Count: {len(wrong_items)}")


def eval_type_pairs_result(
    all_results_path: str,
    wrong_results_path: str,
    strict: bool = False,
    topk: int = -1,
    skip_empty: bool = False
):
    data = []
    with open(all_results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    accs, hits, precs, recs, f1s = [], [], [], [], []
    wrong_items, total = [], 0

    for sample in tqdm(data, desc="Evaluating type_pairs"):
        total += 1
        sid = sample.get("id")
        question = sample.get("question", "")

        gold_list = [tuple(p) for p in (sample.get("ground_type_pairs") or []) if p]
        pred_list = [tuple(p) for p in (sample.get("prediction")   or []) if p]

        if topk and topk > 0 and len(pred_list) > topk:
            pred_list = pred_list[:topk]

        gold_set, pred_set = set(gold_list), set(pred_list)
        if skip_empty and (not gold_set or not pred_set):
            continue

        inter = gold_set & pred_set
        hit = 1 if inter else 0
        acc = 1 if (pred_set == gold_set) else (1 if hit and not strict else 0)

        prec = (len(inter) / len(pred_set)) if pred_set else 0.0
        rec  = (len(inter) / len(gold_set)) if gold_set else 0.0
        f1   = (2*prec*rec / (prec+rec)) if (prec > 0 and rec > 0) else 0.0

        accs.append(acc); hits.append(hit); precs.append(prec); recs.append(rec); f1s.append(f1)

        is_wrong = (acc == 0) if strict else (hit == 0)
        if is_wrong and gold_set:
            wrong_items.append({
                "id": sid,
                "question": question,
                "ground_type_pairs": [list(p) for p in gold_list],
                "prediction":   [list(p) for p in pred_list],
                "intersection": [list(p) for p in sorted(inter)],
                "missing":      [list(p) for p in sorted(gold_set - pred_set)],
                "extra":        [list(p) for p in sorted(pred_set - gold_set)],
                "acc": acc, "hit": hit, "precision": prec, "recall": rec, "f1": f1,
            })

    os.makedirs(os.path.dirname(wrong_results_path), exist_ok=True)
    with open(wrong_results_path, "w", encoding="utf-8") as f:
        for item in wrong_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    print(
        f"Samples: {total} | "
        f"Accuracy: {avg(accs)*100:.2f}  "
        f"Hit: {avg(hits)*100:.2f}  "
        f"F1: {avg(f1s)*100:.2f}  "
        f"Precision: {avg(precs)*100:.2f}  "
        f"Recall: {avg(recs)*100:.2f}"
    )
    print(f"Wrong cases -> {wrong_results_path} | Count: {len(wrong_items)}")

def eval_tail_types_result(
    all_results_path: str,
    wrong_results_path: str,
    strict: bool = False,
    topk: int = -1,
    skip_empty: bool = False
):
    data = []
    with open(all_results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    accs, hits, precs, recs, f1s = [], [], [], [], []
    wrong_items, total = [], 0

    for sample in tqdm(data, desc="Evaluating type_pairs"):
        total += 1
        sid = sample.get("id")
        question = sample.get("question", "")

        gold_list = [
            (p if isinstance(p, str) else tuple(p))
            for p in (sample.get("ground_tail_types") or [])
            if p
        ]
        pred_list = [
            (p if isinstance(p, str) else tuple(p))
            for p in (sample.get("prediction") or [])
            if p
        ]

        if topk and topk > 0 and len(pred_list) > topk:
            pred_list = pred_list[:topk]

        gold_set, pred_set = set(gold_list), set(pred_list)
        if skip_empty and (not gold_set or not pred_set):
            continue

        inter = gold_set & pred_set
        hit = 1 if inter else 0
        acc = 1 if (pred_set == gold_set) else (1 if hit and not strict else 0)

        prec = (len(inter) / len(pred_set)) if pred_set else 0.0
        rec  = (len(inter) / len(gold_set)) if gold_set else 0.0
        f1   = (2*prec*rec / (prec+rec)) if (prec > 0 and rec > 0) else 0.0

        accs.append(acc); hits.append(hit); precs.append(prec); recs.append(rec); f1s.append(f1)

        is_wrong = (acc == 0) if strict else (hit == 0)
        if is_wrong and gold_set:
        # if is_wrong:
            wrong_items.append({
                "id": sid,
                "question": question,
                "ground_tail_types": gold_list,
                "prediction":   pred_list,
                "intersection": [list(p) for p in sorted(inter)],
                "missing":      [list(p) for p in sorted(gold_set - pred_set)],
                "extra":        [list(p) for p in sorted(pred_set - gold_set)],
                "acc": acc, "hit": hit, "precision": prec, "recall": rec, "f1": f1,
            })

    os.makedirs(os.path.dirname(wrong_results_path), exist_ok=True)
    with open(wrong_results_path, "w", encoding="utf-8") as f:
        for item in wrong_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    print(
        f"Samples: {total} | "
        f"Accuracy: {avg(accs)*100:.2f}  "
        f"Hit: {avg(hits)*100:.2f}  "
        f"F1: {avg(f1s)*100:.2f}  "
        f"Precision: {avg(precs)*100:.2f}  "
        f"Recall: {avg(recs)*100:.2f}"
    )
    print(f"Wrong cases -> {wrong_results_path} | Count: {len(wrong_items)}")

def eval_tail_types_result_inf(
    all_results_path: str,
    wrong_results_path: str,
    strict: bool = False,
    topk: int = -1,
    skip_empty: bool = False
):
    def _unique_keep_order(items):
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    data = []
    with open(all_results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    accs, hits, precs, recs, f1s = [], [], [], [], []
    wrong_items, total = [], 0

    for sample in tqdm(data, desc="Evaluating tail types"):
        total += 1
        sid = sample.get("id")
        question = sample.get("question", "")

        raw_gold_paths = sample.get("ground_type_paths") or []
        gold_list = []
        for p in raw_gold_paths:
            if not p:
                continue
            if isinstance(p, str):
                tail = p.split("->")[-1].strip()
            else:
                try:
                    tail = str(p[-1]).strip()
                except Exception:
                    continue
            if tail:
                gold_list.append(tail)

        raw_pred = sample.get("filtered_type_paths") or []
        pred_list = []
        for p in raw_pred:
            if not p:
                continue
            if isinstance(p, str):
                tail = p.split("->")[-1].strip()
            else:
                try:
                    tail = str(p[-1]).strip()
                except Exception:
                    continue
            if tail:
                pred_list.append(tail)

        gold_list = _unique_keep_order(gold_list)
        pred_list = _unique_keep_order(pred_list)

        if topk and topk > 0 and len(pred_list) > topk:
            pred_list = pred_list[:topk]

        gold_set, pred_set = set(gold_list), set(pred_list)
        if skip_empty and (not gold_set or not pred_set):
            continue

        inter = gold_set & pred_set
        hit = 1 if inter else 0
        if strict:
            acc = 1 if (pred_set == gold_set) else 0
        else:
            acc = 1 if hit else 0

        prec = (len(inter) / len(pred_set)) if pred_set else 0.0
        rec  = (len(inter) / len(gold_set)) if gold_set else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec > 0 and rec > 0) else 0.0

        accs.append(acc)
        hits.append(hit)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

        is_wrong = (acc == 0) if strict else (hit == 0)
        if is_wrong and gold_set:
            wrong_items.append({
                "id": sid,
                "question": question,
                "gold_tail_types": list(gold_list),
                "pred_tail_types": list(pred_list),
                "intersection": sorted(list(inter)),
                "missing":      sorted(list(gold_set - pred_set)),
                "extra":        sorted(list(pred_set - gold_set)),
                "acc": acc, "hit": hit, "precision": prec, "recall": rec, "f1": f1,
            })

    os.makedirs(os.path.dirname(wrong_results_path), exist_ok=True)
    with open(wrong_results_path, "w", encoding="utf-8") as f:
        for item in wrong_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    print(
        f"Samples: {total} | "
        f"Accuracy: {avg(accs)*100:.2f}  "
        f"Hit: {avg(hits)*100:.2f}  "
        f"F1: {avg(f1s)*100:.2f}  "
        f"Precision: {avg(precs)*100:.2f}  "
        f"Recall: {avg(recs)*100:.2f}"
    )
    print(f"Wrong cases -> {wrong_results_path} | Count: {len(wrong_items)}")


def extract_topk_prediction(prediction, k=-1):
    if isinstance(prediction, str):
        prediction = prediction.split("\n")
    results = {}
    for p in prediction:
        if p.strip() == "":
            continue
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def _load_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("[") or head.lstrip().startswith("{"):
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                return data
            except json.JSONDecodeError:
                pass

        records = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

def eval_answer_iter_result(all_results_path, predictions_eval_results_path, wrong_results_path,
                       cal_f1=True, topk=-1):
    import os
    import json
    from tqdm import tqdm

    ext = os.path.splitext(all_results_path)[1].lower()
    if ext == ".json":
        with open(all_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
        with open(all_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue

    if not data:
        print("No valid samples to evaluate.")
        return

    base_dir = os.path.dirname(predictions_eval_results_path)

    eval_targets = [
        ("final", "prediction"),
        ("iter1", "generate_answers_iter_1"),
        ("iter2", "generate_answers_iter_2"),
        ("iter3", "generate_answers_iter_3"),
    ]

    all_summary = {}

    def _normalize_answer_field(answer):
        if answer is None:
            return []
        elif isinstance(answer, str):
            return [answer]
        elif isinstance(answer, (list, tuple, set)):
            return list(set(answer))
        else:
            return [str(answer)]

    def _normalize_prediction_field(prediction):
        if prediction is None:
            return []
        elif isinstance(prediction, str):
            return [prediction]
        elif isinstance(prediction, (list, tuple, set)):
            return list(map(str, prediction))
        else:
            return [str(prediction)]

    for tag, pred_field in eval_targets:
        acc_list = []
        hit_list = []
        f1_list = []
        precision_list = []
        recall_list = []

        detailed_results = []
        wrong_results = []

        for sample in tqdm(data, desc=f"Evaluating {tag}"):
            _id = sample.get("id")
            question = sample.get("question")

            prediction = sample.get(pred_field)

            # ground truth
            answer = sample.get("ground_truth")
            if answer is None:
                answer = sample.get("answer")
            answer = _normalize_answer_field(answer)

            if cal_f1:
                prediction = extract_topk_prediction(prediction, topk)
                pred_list = _normalize_prediction_field(prediction)

                if len(answer) == 0:
                    f1_score = precision_score = recall_score = (1.0 if len(pred_list) == 0 else 0.0)
                else:
                    f1_score, precision_score, recall_score = eval_f1(pred_list, answer)

                f1_list.append(f1_score)
                precision_list.append(precision_score)
                recall_list.append(recall_score)

                prediction_str = " ".join(pred_list)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)

                detailed_results.append({
                    "id": _id,
                    "question": question,
                    "eval_target": tag,
                    "prediction_field": pred_field,
                    "prediction": pred_list,
                    "ground_truth": answer,
                    "acc": acc,
                    "hit": hit,
                    "f1": f1_score,
                    "precision": precision_score,
                    "recall": recall_score,
                })

                if hit == 0:
                    wrong_item = dict(sample)
                    wrong_item["eval_target"] = tag
                    wrong_item["prediction_field"] = pred_field
                    wrong_item["evaluated_prediction"] = pred_list
                    wrong_results.append(wrong_item)

            else:
                pred_list = _normalize_prediction_field(prediction)
                pred_str = " ".join(pred_list)

                acc = eval_acc(pred_str, answer)
                hit = eval_hit(pred_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)

                detailed_results.append({
                    "id": _id,
                    "question": question,
                    "eval_target": tag,
                    "prediction_field": pred_field,
                    "prediction": pred_list,
                    "ground_truth": answer,
                    "acc": acc,
                    "hit": hit,
                })

                if hit == 0:
                    wrong_item = dict(sample)
                    wrong_item["eval_target"] = tag
                    wrong_item["prediction_field"] = pred_field
                    wrong_item["evaluated_prediction"] = pred_list
                    wrong_results.append(wrong_item)

        if len(acc_list) == 0:
            print(f"No valid samples to evaluate for {tag}.")
            continue

        summary = {
            "eval_target": tag,
            "prediction_field": pred_field,
            "sample_count": len(acc_list),
            "accuracy": sum(acc_list) * 100 / len(acc_list),
            "hit": sum(hit_list) * 100 / len(hit_list),
        }

        if cal_f1:
            summary["f1"] = sum(f1_list) * 100 / len(f1_list)
            summary["precision"] = sum(precision_list) * 100 / len(precision_list)
            summary["recall"] = sum(recall_list) * 100 / len(recall_list)

        all_summary[tag] = summary

        detail_path = os.path.join(base_dir, f"predictions_eval_results_{tag}.jsonl")
        wrong_path = os.path.join(base_dir, f"wrong_results_{tag}.jsonl")

        with open(detail_path, "w", encoding="utf-8") as f2:
            for item in detailed_results:
                f2.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(wrong_path, "w", encoding="utf-8") as f3:
            for item in wrong_results:
                f3.write(json.dumps(item, ensure_ascii=False) + "\n")

        if cal_f1:
            result_str = (
                f"[{tag}] "
                f"Accuracy: {summary['accuracy']:.2f} "
                f"Hit: {summary['hit']:.2f} "
                f"F1: {summary['f1']:.2f} "
                f"Precision: {summary['precision']:.2f} "
                f"Recall: {summary['recall']:.2f}"
            )
        else:
            result_str = (
                f"[{tag}] "
                f"Accuracy: {summary['accuracy']:.2f} "
                f"Hit: {summary['hit']:.2f}"
            )

        print(result_str)

        result_name = f"eval_result_{tag}_top_{topk}.txt" if topk > 0 else f"eval_result_{tag}.txt"
        eval_result_path = os.path.join(base_dir, result_name)
        with open(eval_result_path, "w", encoding="utf-8") as f:
            f.write(result_str)

    summary_json_path = os.path.join(base_dir, "eval_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    summary_txt_path = os.path.join(base_dir, "eval_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        for tag in ["iter1", "iter2", "iter3", "final"]:
            if tag not in all_summary:
                continue
            s = all_summary[tag]
            if cal_f1:
                line = (
                    f"[{tag}] "
                    f"Accuracy: {s['accuracy']:.2f} "
                    f"Hit: {s['hit']:.2f} "
                    f"F1: {s['f1']:.2f} "
                    f"Precision: {s['precision']:.2f} "
                    f"Recall: {s['recall']:.2f}"
                )
            else:
                line = (
                    f"[{tag}] "
                    f"Accuracy: {s['accuracy']:.2f} "
                    f"Hit: {s['hit']:.2f}"
                )
            f.write(line + "\n")

def eval_answer_result(all_results_path, predictions_eval_results_path, wrong_results_path,
                       cal_f1=True, topk=-1):
    acc_list = []
    hit_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    ext = os.path.splitext(all_results_path)[1].lower()
    if ext == ".json":
        with open(all_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
        with open(all_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue

    detailed_results = []
    wrong_results = []

    for sample in tqdm(data):
        _id = sample.get("id")
        question = sample.get("question")

        prediction = sample.get("prediction")

        answer = sample.get("ground_truth")
        if answer is None:
            answer = sample.get("answer")

        if answer is None:
            answer = []
        elif isinstance(answer, str):
            answer = [answer]
        elif isinstance(answer, (list, tuple, set)):
            answer = list(set(answer))
        else:
            answer = [str(answer)]

        if cal_f1:
            prediction = extract_topk_prediction(prediction, topk)

            if prediction is None:
                pred_list = []
            elif isinstance(prediction, str):
                pred_list = [prediction]
            elif isinstance(prediction, (list, tuple, set)):
                pred_list = list(map(str, prediction))
            else:
                pred_list = [str(prediction)]

            if (len(answer) == 0):
                f1_score = precision_score = recall_score = (1.0 if len(pred_list) == 0 else 0.0)
            else:
                f1_score, precision_score, recall_score = eval_f1(pred_list, answer)

            f1_list.append(f1_score)
            precision_list.append(precision_score)
            recall_list.append(recall_score)

            prediction_str = " ".join(pred_list)
            acc = eval_acc(prediction_str, answer)
            hit = eval_hit(prediction_str, answer)
            acc_list.append(acc)
            hit_list.append(hit)

            detailed_results.append({
                "id": _id,
                "prediction": pred_list,
                "ground_truth": answer,
                "acc": acc,
                "hit": hit,
                "f1": f1_score,
                "precision": precision_score,
                "recall": recall_score,
            })

            if hit == 0:
                wrong_results.append(sample)
        else:
            pred_str = prediction
            if isinstance(prediction, (list, tuple, set)):
                pred_str = " ".join(map(str, prediction))
            elif prediction is None:
                pred_str = ""
            else:
                pred_str = str(prediction)

            acc = eval_acc(pred_str, answer)
            hit = eval_hit(pred_str, answer)
            acc_list.append(acc)
            hit_list.append(hit)

            detailed_results.append({
                "id": _id,
                "prediction": prediction,
                "ground_truth": answer,
                "acc": acc,
                "hit": hit,
            })

            if hit == 0:
                wrong_results.append(sample)

    with open(predictions_eval_results_path, "w", encoding="utf-8") as f2:
        for item in detailed_results:
            f2.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(wrong_results_path, "w", encoding="utf-8") as f3:
        for item in wrong_results:
            f3.write(json.dumps(item, ensure_ascii=False) + "\n")

    if len(acc_list) == 0:
        print("No valid samples to evaluate.")
        return
    result_str = (
        f"Accuracy: {sum(acc_list) * 100 / len(acc_list):.2f} "
        f"Hit: {sum(hit_list) * 100 / len(hit_list):.2f}"
    )

    if cal_f1:
        result_str += (
            f" F1: {sum(f1_list) * 100 / len(f1_list):.2f}"
            f" Precision: {sum(precision_list) * 100 / len(precision_list):.2f}"
            f" Recall: {sum(recall_list) * 100 / len(recall_list):.2f}"
        )

    print(result_str)

    result_name = f"eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = os.path.join(os.path.dirname(predictions_eval_results_path), result_name)
    with open(eval_result_path, "w", encoding="utf-8") as f:
        f.write(result_str)

def eval_result(all_results_path, predictions_eval_results_path, wrong_results_path, cal_f1=True, topk=-1, skip_empty=False):
    acc_sum = hit_sum = 0.0
    f1_sum = precision_sum = recall_sum = 0.0
    n = 0

    acc_list = []
    hit_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    with open(all_results_path, "r", encoding="utf-8") as fin, \
            open(predictions_eval_results_path, "w", encoding="utf-8") as f2, \
            open(wrong_results_path, "w", encoding="utf-8") as f3:
        for line in tqdm(fin, desc="[eval] streaming"):
            line = line.strip()
            if not line:
                continue

            if line.endswith(","):
                line = line[:-1].rstrip()

            if not line.startswith("{") or not line.endswith("}"):
                continue

            try:
                sample = json.loads(line)
            except Exception:
                continue

            id = sample.get("id")
            prediction = sample.get("predictions")
            question = sample.get("question")
            ground_type_pairs = sample.get("ground_type_pairs")

            if prediction is None:
                prediction = sample.get("prediction")

            answer = sample.get("ground_truth")
            if answer is None:
                answer = sample.get("answer")

            if prediction is None:
                prediction = []
            if isinstance(prediction, str):
                prediction = [prediction]
            prediction = [str(x) for x in prediction if str(x).strip()]

            if skip_empty and (not prediction):
                continue

            if cal_f1:
                prediction = extract_topk_prediction(prediction, topk)
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                prediction_str = " ".join(prediction)

                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)

                acc_sum += float(acc)
                hit_sum += float(hit)
                f1_sum += float(f1_score)
                precision_sum += float(precision_score)
                recall_sum += float(recall_score)
                n += 1

                item = {
                    "id": id,
                    "prediction": prediction,
                    "ground_truth": answer,
                    "acc": acc,
                    "hit": hit,
                    "f1": f1_score,
                    "precision": precision_score,
                    "recall": recall_score,
                }
                f2.write(json.dumps(item, ensure_ascii=False) + "\n")

                if hit == 0:
                    f3.write(json.dumps({
                        "id": id,
                        "question": question,
                        "prediction": prediction,
                        "ground_truth": answer,
                        "acc": acc,
                        "hit": hit,
                        "f1": f1_score,
                        "precision": precision_score,
                        "recall": recall_score,
                    }, ensure_ascii=False) + "\n")
            else:
                prediction_str = " ".join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)

                acc_sum += float(acc)
                hit_sum += float(hit)
                n += 1

                item = {"id": id, "prediction": prediction, "ground_truth": answer, "acc": acc, "hit": hit}
                f2.write(json.dumps(item, ensure_ascii=False) + "\n")

                if hit == 0:
                    f3.write(json.dumps({
                        "id": id,
                        "question": question,
                        "prediction": prediction,
                        "ground_truth": answer,
                        "acc": acc,
                        "hit": hit,
                    }, ensure_ascii=False) + "\n")

    if n == 0:
        print("No evaluated samples (n=0).")
        return

    result_str = f"Accuracy: {acc_sum * 100 / n:.2f} Hit: {hit_sum * 100 / n:.2f}"
    if cal_f1:
        result_str += (
            f" F1: {f1_sum * 100 / n:.2f}"
            f" Precision: {precision_sum * 100 / n:.2f}"
            f" Recall: {recall_sum * 100 / n:.2f}"
        )
    print(result_str)

    result_name = f"eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = os.path.join(os.path.dirname(predictions_eval_results_path), result_name)
    with open(eval_result_path, "w", encoding="utf-8") as f:
        f.write(result_str)


def eval_path_result(
    all_results_path: str,
    wrong_results_path: str,
    strict: bool = True,
    topk: int = -1,
    skip_empty: bool = False
):
    data = []
    with open(all_results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    acc_list, hit_list, f1_list, prec_list, rec_list = [], [], [], [], []
    wrong_items = []
    total = 0

    for sample in tqdm(data):
        total += 1
        sid = sample.get("id")
        question = sample.get("question", "")
        q_entity = sample.get("q_entity", [])

        gold_list = list(map(str, (sample.get("ground_type_paths", []) or [])))
        pred_list = list(map(str, (sample.get("filtered_type_paths", []) or [])))

        if topk and topk > 0 and len(pred_list) > topk:
            pred_list = pred_list[:topk]

        if skip_empty and (len(gold_list) == 0 or len(pred_list) == 0):
            continue

        pred_text = ", ".join(pred_list)
        answer_list = gold_list

        acc = eval_acc(pred_text, answer_list)
        hit = eval_hit(pred_text, answer_list)

        f1_score, precision_score, recall_score = eval_f1(pred_list, gold_list)

        acc_list.append(acc)
        hit_list.append(hit)
        prec_list.append(precision_score)
        rec_list.append(recall_score)
        f1_list.append(f1_score)

        if (len(gold_list) > 0) and (hit == 0):
            gold_set = set(gold_list)
            pred_set = set(pred_list)
            wrong_items.append({
                "id": sid,
                "question": question,
                "q_entity": q_entity,
                "ground_type_paths": gold_list,
                "filtered_type_paths": pred_list,
                "acc": acc,
                "hit": hit,
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1_score,
            })

    os.makedirs(os.path.dirname(wrong_results_path), exist_ok=True)
    with open(wrong_results_path, "w", encoding="utf-8") as f:
        for item in wrong_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _avg(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    result_str = (
        f"Samples: {total} | "
        f"Accuracy: {_avg(acc_list) * 100:.2f}  "
        f"Hit: {_avg(hit_list) * 100:.2f}  "
        f"F1: {_avg(f1_list) * 100:.2f}  "
        f"Precision: {_avg(prec_list) * 100:.2f}  "
        f"Recall: {_avg(rec_list) * 100:.2f}"
    )
    print(result_str)
    print(f"Wrong cases excluding empty ground_type_paths -> {wrong_results_path} | Count: {len(wrong_items)}")

