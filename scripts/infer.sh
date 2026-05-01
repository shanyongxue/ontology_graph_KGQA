#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

DATASET=${DATASET:-webqsp}

case "${DATASET}" in
  webqsp)
    DATASET_NAME="RoG-webqsp"
    DATASET_WITH_DATA="RoG-webqsp/data"
    MAX_HOPS=2
    ;;
  cwq)
    DATASET_NAME="RoG-cwq"
    DATASET_WITH_DATA="RoG-cwq/data"
    MAX_HOPS=4
    ;;
  metaqa)
    DATASET_NAME="MetaQA"
    DATASET_WITH_DATA="MetaQA/data"
    MAX_HOPS=4
    ;;
  *)
    echo "[ERROR] Unknown DATASET: ${DATASET}"
    echo "Supported values: webqsp, cwq, metaqa"
    exit 1
    ;;
esac

DATA_ROOT=${DATA_ROOT:-datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
MODEL_ROOT=${MODEL_ROOT:-models}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints}

ONTOLOGY_PATH=${ONTOLOGY_PATH:-${DATA_ROOT}/ontology_triples_general_buquan_final.json}
PROMPT_PATH=${PROMPT_PATH:-prompts/llama2.txt}
SBERT_MODEL=${SBERT_MODEL:-${MODEL_ROOT}/all-mpnet-base-v2}
TAIL_MODEL_DIR=${TAIL_MODEL_DIR:-${CHECKPOINT_ROOT}/tail_type_model_${DATASET}}
TAIL_MODEL_NAME=${TAIL_MODEL_NAME:-tail_type_model_${DATASET}}

TAIL_PRED_ROOT="${OUTPUT_ROOT}/tail_type_predictions"
TAIL_PRED_PATH="${TAIL_PRED_ROOT}/${DATASET_WITH_DATA}/${TAIL_MODEL_NAME}/test/predictions_1_False.jsonl"

RETRIEVAL_DIR="${OUTPUT_ROOT}/retrieval/${DATASET_NAME}"
RETRIEVAL_PATH="${RETRIEVAL_DIR}/predictions_from_filtered_type_paths.jsonl"

FINAL_OUTPUT_ROOT="${OUTPUT_ROOT}/final_predictions"

mkdir -p "${TAIL_PRED_ROOT}" "${RETRIEVAL_DIR}" "${FINAL_OUTPUT_ROOT}"

echo "[1/3] Predicting answer tail types..."
python src/qa_prediction/gen_tail_types.py \
  --data_path "${DATA_ROOT}" \
  --d "${DATASET_WITH_DATA}" \
  --split test \
  --ontology_path "${ONTOLOGY_PATH}" \
  --output_path "${TAIL_PRED_ROOT}" \
  --model_name "${TAIL_MODEL_NAME}" \
  --model_path "${TAIL_MODEL_DIR}" \
  --prompt_path "${PROMPT_PATH}" \
  --top_k 1 \
  --type_sim_threshold 0.8 \
  --max_new_tokens 100 \
  --n_beam 1 \
  --max_length "${MAX_HOPS}" \
  --force

echo "[2/3] Running ontology-guided bidirectional retrieval..."
python src/qa_prediction/bidirectional_retrieval.py \
  --dataset_path "${DATA_ROOT}/${DATASET_NAME}" \
  --train_split train \
  --test_split test \
  --tailtype_path_test "${TAIL_PRED_PATH}" \
  --output_path "${RETRIEVAL_PATH}" \
  --ontology_triples_path "${ONTOLOGY_PATH}" \
  --max_reasoning_paths 256 \
  --path_rank_model_path "${SBERT_MODEL}" \
  --max_hops "${MAX_HOPS}"

echo "[3/3] Running iterative answer refinement..."
python src/qa_prediction/iterative_answer_refinement.py \
  --data_path "${DATA_ROOT}" \
  --ontology_path "${ONTOLOGY_PATH}" \
  --d "${DATASET_NAME}" \
  --split test \
  --predict_path "${FINAL_OUTPUT_ROOT}" \
  --rule_path "${RETRIEVAL_PATH}" \
  --openai_api_key "${OPENAI_API_KEY:-}" \
  --openai_base_url "${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
  --openai_model "${OPENAI_MODEL:-gpt-4o}" \
  --openai_temperature 0.2 \
  --openai_max_tokens 128 \
  --max_return 5 \
  --mpnet_path "${SBERT_MODEL}" \
  --force

echo "[DONE] Inference finished. Results saved under: ${FINAL_OUTPUT_ROOT}"