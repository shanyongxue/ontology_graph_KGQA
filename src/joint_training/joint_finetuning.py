import sys
import os
import time
import torch
import warnings
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from utils import *
import logging
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import AutoPeftModelForCausalLM, LoraConfig
from datasets import load_dataset, concatenate_datasets, Dataset

import datasets
datasets.disable_progress_bar()

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1
torch.cuda.empty_cache()

def load_new_tokens(default_new_tokens, rel_dict_path):
    if isinstance(rel_dict_path, str):
        rel_dict_path = [rel_dict_path]
    for rel_path in rel_dict_path:
        with open(rel_path, 'r') as f:
            for line in f.readlines():
                _, r = line.strip().split('\t')
                default_new_tokens.append(r)
    return default_new_tokens
        

def load_multiple_datasets(data_path_list, shuffle=False):
    '''
    Load multiple datasets from different paths.

    Args:
        data_path_list (_type_): _description_
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    '''
    dataset_list = [load_dataset('json', data_files=p, split="train")
                     for p in data_path_list]
    dataset = concatenate_datasets(dataset_list)
    if shuffle:
        dataset = dataset.shuffle()
    return dataset

def smart_tokenizer_and_embedding_resize(
    new_tokens,
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_tokens(new_tokens)
    num_new_special_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    total_new_tokens = num_new_tokens + num_new_special_tokens
    if total_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

@dataclass
class ScriptArguments:
    data_path_list: list[str] = field(
        metadata={"help": "Path to the training data."}
    )
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    rel_dict_path: list[str] = field(
        default=None, metadata={"help": "Path to the relation dictionary."}
    )
    add_rel_token: Optional[bool] = field(
        default=False, metadata={"help": "Wether to add relation token or not"}
    )
    prompt_path: str = field(
        default="prompts/llama2.txt",
        metadata={"help": "Path to the prompt template"},
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Wether to use PEFT or not to train adapters"},
    )
    save_merged: Optional[bool] = field(
        default=False, metadata={"help": "Wether to save merged model"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(
        default=8, metadata={"help": "the lora r parameter"}
    )
    load_in_4bit: bool = field(default=False, metadata={"help": "Load model in 4bit"})
    load_in_8bit: bool = field(default=False, metadata={"help": "Load model in 8bit"})
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "attn implementation"})
    response_template: Optional[str] = field(default="[INST]", metadata={"help": "Response template"})


@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="saved_models/llama2_align",
        metadata={"help": "The output directory"},
    )
    optim: str = field(default="adamw_torch")
    max_seq_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    ddp_find_unused_parameters: bool = field(default=False)

def train():
    parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load models
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        # token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        attn_implementation=script_args.attn_implementation,
        load_in_4bit=script_args.load_in_4bit,
        load_in_8bit=script_args.load_in_8bit,
        # device=torch.device("cuda:1"),
        # device_map={"": Accelerator().local_process_index},
    )

    model.config.use_cache = False
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            # target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        # token=HF_TOKEN,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = '<PAD>'
    new_tokens = ['<SEP>', '<PAIR>', '</PAIR>', '<ANSWER>', '</ANSWER>', '<JUDGE>', '</JUDGE>']
    if script_args.add_rel_token:
        new_tokens = load_new_tokens(new_tokens, script_args.rel_dict_path)
    smart_tokenizer_and_embedding_resize(new_tokens, special_tokens_dict, tokenizer, model)
    
    tokenizer.padding_side = "right"

    # Load datasets
    train_dataset = load_multiple_datasets(script_args.data_path_list, shuffle=True)
    
    response_template = script_args.response_template
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

    sft_cfg = SFTConfig(
        **training_args.to_dict(),
        dataset_text_field="text",
        packing=False,
        dataset_kwargs={"add_special_tokens": False},
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=sft_cfg,
        data_collator=data_collator,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
                last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    if script_args.use_peft:
        trainer.model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        if script_args.save_merged:
            del model
            torch.cuda.empty_cache()
            model = AutoPeftModelForCausalLM.from_pretrained(
                training_args.output_dir, device_map="auto", torch_dtype=torch.bfloat16
            )
            model = model.merge_and_unload()
            model.eval()
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()
    print(f"Time: {end_time - start_time:.2f} seconds.")