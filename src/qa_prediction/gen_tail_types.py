import json
import sys
import os
import argparse
import utils
from datasets import load_dataset
import datasets

datasets.disable_progress_bar()
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
import re
from utils.qa_utils import eval_tail_types_result
from utils.graph_utils import get_tail_types_from_relations
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Dict

N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)
TAIL_RE = r"<TAIL>(.*)<\/TAIL>"
INSTRUCTION="""<TASK:ANSWER_TYPES>\nPlease generate a valid answer entity type that can be helpful for answering the following question: """

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results

def parse_prediction(prediction):
    """
    Parse a list of predictions to a flat list of types

    Args:
        prediction (List[str]): List of raw prediction strings e.g., ["<TAIL>type</TAIL>"]

    Returns:
        List[str]: Flat list of types e.g., ["type1", "type2"]
    """
    results = []
    for p in prediction:
        # 1. Extract the content between <TAIL> and </TAIL>.
        content = re.search(TAIL_RE, p)
        if content is None:
            continue
        content = content.group(1)

        # 2. Split by <SEP> if multiple tail types are generated.
        parts = content.split("<SEP>")

        # 3. Clean and append each tail type to the flat result list.
        for part in parts:
            part = part.strip()
            if part:
                results.append(part)

    return results


def generate_seq(
    model, input_text, tokenizer, num_beam=3, do_sample=False, max_new_tokens=100
):
    # tokenize the question
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    # generate sequences
    output = model.generate(
        input_ids=input_ids,
        num_beams=num_beam,
        num_return_sequences=num_beam,
        early_stopping=False,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
    )
    prediction = tokenizer.batch_decode(
        output.sequences[:, input_ids.shape[1] :], skip_special_tokens=True
    )
    prediction = [p.strip() for p in prediction]

    if num_beam > 1:
        scores = output.sequences_scores.tolist()
        norm_scores = torch.softmax(output.sequences_scores, dim=0).tolist()
    else:
        scores = [1]
        norm_scores = [1]

    return {"tails": prediction, "scores": scores, "norm_scores": norm_scores}

def load_ontology_triples(path):
    with open(path, 'r') as f:
        triples = json.load(f)
    return triples

def gen_prediction(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if args.lora or os.path.exists(args.model_path + "/adapter_config.json"):
        print("Load LORA model")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=True,
        )

    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d, args.model_name, args.split)
    print("Save results to: ", output_dir)
    print("input files", input_file)

    # Load dataset
    dataset = load_dataset(input_file, split=args.split)

    ontology_triples = load_ontology_triples(args.ontology_path)
    rel2edges = defaultdict(set)
    for h, r, t in ontology_triples:
        rel2edges[r].add((h, t))

    # Load prompt template
    prompter = utils.InstructFormater(args.prompt_path)

    def prepare_dataset(sample, rel2edges, prompter=None, include_prompt=True, remove_duplicate=True):
        """
        Build ground_tail_types for a single sample:
        1) Strictly compose all shortest relation paths hop by hop and take their union.
        2) If strict composition fails, fall back to tail types inferred from the last relation.
        """
        if include_prompt and prompter is not None:
            sample["text"] = prompter.format(instruction=INSTRUCTION, message=sample["question"])

        # 1) Find shortest entity paths and extract deduplicated non-empty relation paths.
        graph = utils.build_graph_new(sample["graph"])
        paths = utils.get_truth_paths_new(sample["q_entity"], sample["a_entity"], graph, max_hops=args.max_length)

        rel_paths = []
        seen_rel_paths = set()
        for path in paths:
            rel_path = tuple(p[1] for p in path)  # ('r1','r2',...)
            if not rel_path:
                continue
            if remove_duplicate and rel_path in seen_rel_paths:
                continue
            rel_paths.append(rel_path)
            seen_rel_paths.add(rel_path)

        sample["ground_paths"] = [list(p) for p in seen_rel_paths]

        # 2) Strictly compose all relation paths hop by hop and take the union.
        strict_tails_all = set()
        for rel_path in rel_paths:
            strict_tails_all |= get_tail_types_from_relations(list(rel_path), rel2edges)

        result_tails = set(strict_tails_all)

        # 3) If strict composition produces no result, fall back to tail types from the last relation.
        if not result_tails and rel_paths:
            tails_union = set()
            for rel_path in rel_paths:
                last_rel = rel_path[-1]
                tails_union |= {t for _, t in rel2edges.get(last_rel, set())}
            if tails_union:
                result_tails = tails_union

        # 4) Save ground tail types. If result_tails is still empty, save an empty list.
        sample["ground_tail_types"] = list(result_tails)

        return sample

    dataset = dataset.map(
        prepare_dataset,
        fn_kwargs={
            "rel2edges": rel2edges,
            "prompter": prompter,
            "include_prompt": True,
            "remove_duplicate": True,
        },
        num_proc=N_CPUS,
    )

    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_file = os.path.join(output_dir, f"predictions_{args.n_beam}_{args.do_sample}.jsonl")
    f, processed_results = get_output_file(prediction_file, force=args.force)
    for data in tqdm(dataset):
        question = data["question"]
        input_text = data["text"]
        qid = data["id"]
        if qid in processed_results:
            continue
        raw_output = generate_seq(
            model,
            input_text,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            num_beam=args.n_beam,
            do_sample=args.do_sample,
        )
        predict_tails = parse_prediction(raw_output["tails"])
        if args.debug:
            print("ID: ", qid)
            print("Question: ", question)
            print("Prediction: ", predict_tails)
        # prediction = outputs[0]["generated_text"].strip()
        data = {
            "id": qid,
            "question": question,
            "prediction": list(predict_tails),
            "ground_tail_types": data["ground_tail_types"],
            "input": input_text,
            "raw_output": raw_output,
        }
        f.write(json.dumps(data) + "\n")
        f.flush()
    f.close()

    # Evaluate predictions and save wrong cases.
    base_dir = os.path.dirname(prediction_file)
    wrong_results_path = os.path.join(base_dir, "wrong_results.jsonl")
    eval_tail_types_result(prediction_file, wrong_results_path)

    wrong_results_analyze_path = os.path.join(base_dir, "wrong_results_analyze.json")
    wrong_items = []
    if os.path.exists(wrong_results_path):
        with open(wrong_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    wrong_items.append(json.loads(line))
    wrong_ids = set(item.get("id", "") for item in wrong_items)
    matched_samples = [sample for sample in dataset if sample.get("id", "") in wrong_ids]
    with open(wrong_results_analyze_path, "w", encoding="utf-8") as f:
        for item in matched_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Extracted {len(matched_samples)} original samples with wrong predictions to {wrong_results_analyze_path}")


    return prediction_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="datasets")
    parser.add_argument("--d", "-d", type=str, default="RoG-webqsp/data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ontology_path", type=str, default="datasets/ontology_triples_general_buquan_final.json")
    parser.add_argument("--output_path", type=str, default="outputs/tail_type_predictions")
    parser.add_argument("--model_name", type=str, default="tail_type_model", help="Model name used for saving results.")
    parser.add_argument("--model_path", type=str, default="checkpoints/tail_type_model",
                        help="Path to the tail-type prediction model.")
    parser.add_argument("--prompt_path", type=str, default="prompts/llama2.txt")
    parser.add_argument("--rel_dict", nargs="+", default=[], help="Optional relation dictionary.")
    parser.add_argument("--force", "-f", action="store_true", default=False, help="Overwrite existing results.")
    parser.add_argument("--ontology_emb_path", type=str, default="", help="Optional ontology embedding path.")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--type_sim_threshold", type=float, default=0.8)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lora", action="store_true", help="Load LoRA weights.")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--n_beam", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--max_length", "-max_path_length", type=int, default=2)

    args = parser.parse_args()

    gen_path = gen_prediction(args)
