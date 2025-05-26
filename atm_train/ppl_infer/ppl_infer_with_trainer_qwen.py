import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from dataclasses import dataclass
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from accelerate import Accelerator
from transformers.trainer_utils import is_main_process

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


class LossPerSampleTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs['labels']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            losses = loss_fct(shift_logits.transpose(1,2), shift_labels)
            per_sample_loss = (losses.sum(dim=1) / (shift_labels != -100).sum(dim=1))
            per_sample_loss = per_sample_loss.float()
            return (None, per_sample_loss, None)  # logits -> results.predictions




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
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--input_file", required=True)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--num_dups", type=int, default=5)
    p.add_argument("--output", required=True)
    return p.parse_args()

def main():
    accelerator = Accelerator()
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)

    # 1) 处理成一条一条 prompt
    with accelerator.main_process_first():
        raw_ds = load_dataset("json", data_files=args.input_file, split="train")
        flat_ds = raw_ds.map(build_prompt_records, batched=True, remove_columns=raw_ds.column_names)
        proc_ds = flat_ds.map(lambda ex: tokenize(ex, tokenizer), batched=False, remove_columns=flat_ds.column_names)
        print(f'args.num_dups: {args.num_dups}')
        print(f'proc_ds: {proc_ds}')
    # return
    # 2) Trainer + DataCollator
    @dataclass
    class DataCollatorWithLabelPad:
        tokenizer: Any
        pad_to_multiple_of: int = 8
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            labels = [f.pop("labels") for f in features]
            batch = self.tokenizer.pad(
                features,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            max_len = batch["input_ids"].shape[1]
            for i, l in enumerate(labels):
                labels[i] = l + [-100]*(max_len - len(l))
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
            return batch

    training_args = TrainingArguments(
        output_dir="./eval_outdir",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        remove_unused_columns=False,
        report_to="none",
        prediction_loss_only=True,
    )
    trainer = LossPerSampleTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithLabelPad(tokenizer, pad_to_multiple_of=8),
        tokenizer=tokenizer,
        eval_dataset=proc_ds,
    )

    results = trainer.predict(proc_ds)
    accelerator.wait_for_everyone()


    pred = results.predictions[:proc_ds.num_rows].reshape((-1, args.num_dups))
    # ppl_per_sample = np.exp(predictions)
    if trainer.is_world_process_zero():
        odf = pd.DataFrame(pred, columns=[f'output_{idx}' for idx in range(args.num_dups)])
        odf.to_csv(f'{args.output}', index=False)
        print("保存每条样本的loss/ppl至", args.output)


if __name__ == "__main__":
    main()
