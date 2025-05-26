
import logging
import sys
import os
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompting_for_rag import get_prompt
from accelerate import Accelerator
import pickle

from modeling_ppllama import LlamaPPL
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaTokenizer,
    DataCollatorForSeq2Seq, 
    set_seed,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from torch.utils.data import Subset
import re
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser
from tqdm.std import *
import numpy as np


MAX_LENGTH = 2048

def build_prompt_records(batch):
    NUM_DUPS = 5
    inputs, targets = [], []
    for ex_idx in range(len(batch["question"])):
        for ctx in batch["ctxs"][ex_idx][:NUM_DUPS]:
            paragraph = f"[document] # Title: {ctx['title']} ## text: {ctx['text']} [/document]"
            prompt = (
                "You are a helpful assistant. Below is a question and some retrieved documents (some may be irrelevant).\n"
                "Use them to write a high-quality, concise, and accurate answer.\n\n"
                "[Knowledge]\n"
                f"{paragraph}\n"
                f"Question: {batch['question'][ex_idx]}\n\nAnswer:\n"
            )
            inputs.append(prompt)
            targets.append(batch["answers"][ex_idx][0])
    return {"input": inputs, "target": targets}

def tokenize(example, tokenizer):
    enc = tokenizer(example["input"], truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
    ans = tokenizer(example["target"], truncation=True, max_length=MAX_LENGTH//4, add_special_tokens=False)
    input_ids = enc["input_ids"] + ans["input_ids"] + [tokenizer.eos_token_id]
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = [1] * len(input_ids)
    labels = [-100]*len(enc["input_ids"]) + ans["input_ids"] + [tokenizer.eos_token_id]
    labels = labels[:len(input_ids)]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def parse_args():
    parser = argparse.ArgumentParser(description="PPL ranker")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True)
    parser.add_argument("--num_dups", type=int,default=5)
    parser.add_argument("--output", type=str, required=True)

    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = LlamaPPL.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    ds = load_dataset('json', data_files=args.input_file, split='train')
    
    with accelerator.main_process_first():
        ds = ds.map(build_prompt_records, batched=True, remove_columns=ds.column_names)
        ds = ds.map(lambda ex: tokenize(ex, tokenizer), batched=False, remove_columns=ds.column_names)
        print(ds)


    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval_outdir",
        save_strategy = "no",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        tokenizer=tokenizer,
        eval_dataset=ds
    )

    preds = trainer.predict(ds)
    accelerator.wait_for_everyone()
    
    preds = preds.predictions[:ds.num_rows].reshape((-1, args.num_dups))
    odf = pd.DataFrame(preds, columns=[f'output_{idx}' for idx in range(args.num_dups)])
    odf.to_csv(f'{args.output}', index=False)

if __name__ == "__main__":
    NUM_DUPS = 5
# NUM_DUPS = 10
    main()

