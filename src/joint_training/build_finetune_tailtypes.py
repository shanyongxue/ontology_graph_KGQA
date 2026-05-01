"""Convert question-tail-type pairs into instruction-tuning examples."""

import json
import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
from utils.utils import InstructFormater
from transformers import AutoTokenizer
import time

def build_answer_prompt(question: str,
                           prompter: InstructFormater,
                           eos_token: str = "") -> str:
    instruction = (
        "Given a question, please generate a valid Freebase-style answer entity type that help answer the question."
        "Output ONLY the type string in dot-separated format."
    )

    QUESTION = "\nQuestion:{question}\n"
    message = QUESTION.format(question=question)
    return prompter.format(instruction=instruction, message=message)

def build_llama_finetune_data(tail_types_path: str, output_path: str,
                              prompter: InstructFormater, eos_token: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = load_dataset("json", data_files=tail_types_path, split="train")

    with open(output_path, "w", encoding="utf-8") as output_f:
        skipped = 0

        for sample in tqdm(dataset):
            question = (sample.get("question") or "").strip()
            tail_type = sample.get("tail_type", [])

            if isinstance(tail_type, str):
                tail_type_string = tail_type.strip()
            elif isinstance(tail_type, (list, tuple)):
                if len(tail_type) == 1:
                    tail_type_string = str(tail_type[0]).strip()
                else:
                    sep_token = "<SEP>"
                    flat = []
                    for x in tail_type:
                        if isinstance(x, (list, tuple)):
                            flat.extend(map(lambda s: str(s).strip(), x))
                        else:
                            flat.append(str(x).strip())
                    tail_type_string = sep_token.join([s for s in flat if s])
            else:
                tail_type_string = str(tail_type).strip()

            if not question or not tail_type_string:
                skipped += 1
                continue

            prompt_text = build_answer_prompt(
                question=question,
                prompter=prompter,
                eos_token=eos_token,
            )

            text = f"{prompt_text} <TAIL>{tail_type_string}</TAIL>{eos_token}"
            output_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    output_f.close()
    print(f"Saved instruction-tuning data to: {output_path}")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="datasets/RoG-webqsp")
    parser.add_argument("--tail_types_path", default="outputs/tail_type_raw/RoG-webqsp/tail_types_dataset_train.jsonl")
    parser.add_argument("--prompt_path", default="prompts/llama2.txt")
    parser.add_argument("--model_path", default="models/Llama-2-7b-chat-hf")
    parser.add_argument("--output_path",
                        default="outputs/finetune_tail_types/RoG-webqsp/finetune_question2tailtypes_train.jsonl")
    args = parser.parse_args()

    prompter = InstructFormater(args.prompt_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    eos_token = tokenizer.eos_token or ""

    build_llama_finetune_data(args.tail_types_path, args.output_path, prompter, eos_token)

    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f} seconds.")