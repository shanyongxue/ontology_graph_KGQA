# ORBIT

This repository provides the implementation of **ORBIT**, an ontology-guided framework for multi-hop knowledge graph question answering. ORBIT first predicts answer-side entity types, then performs ontology-guided bidirectional retrieval to collect candidate reasoning paths, and finally applies iterative answer refinement to generate final answers.

## Overview

ORBIT consists of three main components:

1. **Tail-type prediction**: fine-tunes a language model to predict answer-side Freebase-style entity types for each question.
2. **Ontology-guided bidirectional retrieval**: uses predicted answer types and ontology relation signatures to retrieve compact reasoning evidence from the knowledge graph.
3. **Iterative answer refinement**: uses a generator-refiner loop to produce and revise final answers based on retrieved reasoning paths, candidate answers, and type constraints.

The full pipeline is provided through two scripts:

```text
scripts/train.sh
scripts/infer.sh
```

## Repository Structure

```text
.
├── config/
│   └── deepspeed_zero3_gcr.yml
├── datasets/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── prompts/
│   ├── general_prompt.txt
│   ├── llama2.txt
│   ├── llama2_predict.txt
│   └── qwen2.txt
├── scripts/
│   ├── train.sh
│   └── infer.sh
├── src/
│   ├── joint_training/
│   │   ├── build_tail_types_dataset_reconstruct.py
│   │   ├── build_finetune_tailtypes.py
│   │   └── joint_finetuning.py
│   ├── qa_prediction/
│   │   ├── gen_tail_types.py
│   │   ├── bidirectional_retrieval.py
│   │   └── iterative_answer_refinement.py
│   └── utils/
│       ├── graph_utils.py
│       ├── qa_utils.py
│       ├── training_utils.py
│       └── utils.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Generated files such as datasets, model checkpoints, and prediction outputs are not tracked by Git.

## Installation

Create a Python environment:

```bash
conda create -n orbit python=3.10 -y
conda activate orbit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, make sure your local PyTorch, CUDA, and DeepSpeed versions are compatible with your hardware.

## Data Preparation

Place datasets under `datasets/`.

The expected dataset layout is:

```text
datasets/
├── RoG-webqsp/
│   └── data/
├── RoG-cwq/
│   └── data/
├── MetaQA/
│   └── data/
└── ontology_triples_general_buquan_final.json
```

Each dataset should be loadable by Hugging Face `datasets.load_dataset`.

The ontology file should be a JSON list of triples:

```json
[
  ["head_type", "relation", "tail_type"]
]
```

## Model Preparation

Place the base LLM and Sentence-BERT model under `models/`.

Example:

```text
models/
├── Llama-2-7b-chat-hf/
└── all-mpnet-base-v2/
```

The default training script assumes the base model path is:

```text
models/Llama-2-7b-chat-hf
```

The default path-ranking model is:

```text
models/all-mpnet-base-v2
```

You can override these paths through environment variables.

## Training

Run tail-type model training with:

```bash
DATASET=webqsp bash scripts/train.sh
```

Supported dataset names are:

```bash
DATASET=webqsp
DATASET=cwq
DATASET=metaqa
```

The training script performs three steps:

1. build tail-type supervision data;
2. convert supervision data into LLaMA-style instruction-tuning data;
3. fine-tune the tail-type prediction model.

Important configurable variables:

```bash
DATASET=webqsp
DATA_ROOT=datasets
OUTPUT_ROOT=outputs
MODEL_ROOT=models
CHECKPOINT_ROOT=checkpoints
ONTOLOGY_PATH=datasets/ontology_triples_general_buquan_final.json
BASE_MODEL=models/Llama-2-7b-chat-hf
PROMPT_PATH=prompts/llama2.txt
TRAIN_CONFIG=config/deepspeed_zero3_gcr.yml
TAIL_MODEL_DIR=checkpoints/tail_type_model_webqsp
BATCH_SIZE=4
EPOCH=3
GRADIENT_ACCUMULATION_STEPS=16
```

Example with custom paths:

```bash
DATASET=webqsp \
BASE_MODEL=models/Llama-2-7b-chat-hf \
TAIL_MODEL_DIR=checkpoints/tail_type_model_webqsp \
bash scripts/train.sh
```

The trained model will be saved to:

```text
checkpoints/tail_type_model_<dataset>
```

## Inference

Before running inference, configure an OpenAI-compatible API endpoint.

You can either export environment variables:

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini
```

or create a local `.env` file based on `.env.example`.

Run the full inference pipeline:

```bash
DATASET=webqsp \
TAIL_MODEL_DIR=checkpoints/tail_type_model_webqsp \
bash scripts/infer.sh
```

The inference script performs three steps:

1. predict answer tail types;
2. run ontology-guided bidirectional retrieval;
3. run iterative answer refinement.

Important configurable variables:

```bash
DATASET=webqsp
DATA_ROOT=datasets
OUTPUT_ROOT=outputs
MODEL_ROOT=models
CHECKPOINT_ROOT=checkpoints
ONTOLOGY_PATH=datasets/ontology_triples_general_buquan_final.json
PROMPT_PATH=prompts/llama2.txt
SBERT_MODEL=models/all-mpnet-base-v2
TAIL_MODEL_DIR=checkpoints/tail_type_model_webqsp
TAIL_MODEL_NAME=tail_type_model_webqsp
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

Final prediction results are saved under:

```text
outputs/final_predictions/
```

Intermediate outputs are saved under:

```text
outputs/tail_type_predictions/
outputs/retrieval/
```

## Output Files

After inference, the main output directory contains:

```text
predictions.jsonl
eval_summary.json
eval_metrics_all.txt
wrong_results.json
```

The `predictions.jsonl` file stores final predictions and intermediate iteration results.

The `eval_summary.json` file stores evaluation metrics.

The `wrong_results.json` file stores incorrectly answered samples for error analysis.

## Environment Variables

The scripts are designed to be configured through environment variables rather than editing the source code.

Common examples:

```bash
DATASET=webqsp
DATA_ROOT=datasets
OUTPUT_ROOT=outputs
MODEL_ROOT=models
CHECKPOINT_ROOT=checkpoints
BASE_MODEL=models/Llama-2-7b-chat-hf
TAIL_MODEL_DIR=checkpoints/tail_type_model_webqsp
SBERT_MODEL=models/all-mpnet-base-v2
OPENAI_MODEL=gpt-4o-mini
```


## Quick Start

### Optional: Training

### Run inference:

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o

DATASET=webqsp \
TAIL_MODEL_DIR=checkpoints/tail_type_model_webqsp \
bash scripts/infer.sh
```