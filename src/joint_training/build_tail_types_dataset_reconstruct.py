"""Build supervision data for answer tail-type prediction.

For each question, this script extracts relation paths between topic entities and gold answer entities.
"""

import sys
import os

import argparse
import os
import json
from datasets import load_dataset
import multiprocessing as mp
import utils
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from utils.graph_utils import get_tail_types_from_relations

# Load ontology triples and match the start and end types of relation paths.
def load_ontology_triples(path):
    with open(path, 'r') as f:
        triples = json.load(f)
    return triples

def extract_tail_types(raw_values):
    """
    Convert the output of get_tail_types_from_relations into a set of tail entity type strings.
    """
    tail_types = set()
    for v in raw_values:
        if isinstance(v, (tuple, list)) and len(v) > 0:
            tail_types.add(v[-1])
        else:
            tail_types.add(v)
    return tail_types
def build_data(args):
    '''
    Extract the paths between question and answer entities from the dataset.
    '''

    input_file = os.path.join(args.data_path, args.d + "/data")
    output_dir = os.path.join(args.output_path, args.d)

    print("Save results to: ", output_dir)
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    # dataset = dataset.select(range(20))

    ontology_triples = load_ontology_triples(args.ontology_path)
    rel2edges = defaultdict(set)
    for h, r, t in ontology_triples:
        rel2edges[r].add((h, t))

    with open(os.path.join(output_dir, args.save_name), 'w') as fout:
        for data in tqdm(dataset, total=len(dataset)):
            results = process_data(data, rel2edges, remove_duplicate=args.remove_duplicate)
            for r in results:
                fout.write(json.dumps(r) + '\n')


def process_data(data, rel2edges, remove_duplicate=False):
    question = data['question']
    graph = utils.build_graph_new(data['graph'])
    paths = utils.get_truth_paths_new(data['q_entity'], data['a_entity'], graph)

    # Extract relation paths and optionally remove duplicates.
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

    strict_pairs_all = set()
    for rel_path in rel_paths:
        strict_pairs_all |= get_tail_types_from_relations(list(rel_path), rel2edges)

    tail_types = extract_tail_types(strict_pairs_all)

    if not tail_types:
        non_empty_rel_paths = [rp for rp in rel_paths if rp]
        if non_empty_rel_paths:
            for rel_path in non_empty_rel_paths:
                first_rel, last_rel = rel_path[0], rel_path[-1]
                tail_types |= {t for _, t in rel2edges.get(last_rel, set())}

    samples = [{"question": question, "tail_type": tp} for tp in sorted(tail_types)]
    if len(strict_pairs_all)==0:
        print("question:{}, num_strict_tail_types: {}, num_final_tail_types: {}".format(question, len(strict_pairs_all), len(tail_types)))
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="datasets")
    parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ontology_path", type=str, default="datasets/ontology_triples_general_buquan_final.json")
    parser.add_argument("--output_path", type=str, default="outputs/tail_type_raw")
    parser.add_argument("--save_name", type=str, default="tail_types_dataset_train.jsonl")
    parser.add_argument('--n', '-n', type=int, default=1)
    parser.add_argument('--remove_duplicate', action='store_true', default=False)
    args = parser.parse_args()

    if args.save_name == "":
        args.save_name = args.d + "_" + args.split + ".jsonl"

    build_data(args)
