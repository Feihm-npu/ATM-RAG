import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from dataclasses import dataclass
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from accelerate import Accelerator
from transformers.trainer_utils import is_main_process

MAX_LENGTH = 2048




def template_from_file(example):
    NUM_DUPS = 5
    item = {}

    item['answer'] = example['answers'][0]
    item['question'] = example['question']

    example['input'] = []
    for ctx in example["ctxs"][:NUM_DUPS]:
        paragraph = f"[document] # Title: {ctx['title']} ## text: {ctx['text']} [/document]"
        prompt = (
                "You are a helpful assistant. Below is a question and some retrieved documents (some may be irrelevant).\n"
                "Use them to write a high-quality, concise, and accurate answer.\n\n"
                "[Knowledge]\n"
                f"{paragraph}\n"
                f"Question: {item['question']}\n\nAnswer:\n"
            )
        example['input'].append(prompt)

    example['target'] = [example['answers'][0] for _ in range(NUM_DUPS)]
    
    return example


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




def format_tokenize_row(example, tokenizer):
    assert tokenizer.padding_side == 'left'
    input_ = example['input'][0]
    target = example['target'][0]


    encs = tokenizer(input_, padding=True, add_special_tokens=False)
    example['input_ids'] = encs['input_ids']
    example['attention_mask'] = encs['attention_mask']
    
    ans_encs = tokenizer(target, add_special_tokens=False)
    
    example['labels'] = [[-100] * len(row_enc) for row_enc in example['input_ids']]
    

    for idx, item in enumerate(example['labels']):
        example['input_ids'][idx] += (ans_encs['input_ids'][idx] + [tokenizer.eos_token_id])
        example['labels'][idx] += (ans_encs['input_ids'][idx] + [tokenizer.eos_token_id])
        example['attention_mask'][idx] += [1] * len(ans_encs['input_ids'][idx] + [tokenizer.eos_token_id])
        assert len(example['input_ids'][idx]) == len(example['labels'][idx])
        assert len(example['attention_mask'][idx]) == len(example['labels'][idx])



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--input_file", required=True)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--num_procs", type=int, default=8)
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
    ds = load_dataset("json", data_files=args.input_file, split="train")
    # 1) 处理成一条一条 prompt
    with accelerator.main_process_first():
        ds = ds.map(template_from_file, num_proc=args.num_procs, remove_columns=ds.column_names)
        ds = ds.map(format_tokenize_row, fn_kwargs={'tokenizer': tokenizer}, num_proc=args.num_proc, remove_columns=ds.column_names, batched=True, batch_size=1)
        print(ds)
    # return
    # 2) Trainer + DataCollator


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
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        tokenizer=tokenizer,
        eval_dataset=ds,
    )

    results = trainer.predict(ds)
    accelerator.wait_for_everyone()


    pred = results.predictions[:ds.num_rows].reshape((-1, args.num_dups))
    # ppl_per_sample = np.exp(predictions)
    if trainer.is_world_process_zero():
        odf = pd.DataFrame(pred, columns=[f'output_{idx}' for idx in range(args.num_dups)])
        odf.to_csv(f'{args.output}', index=False)
        print("保存每条样本的loss/ppl至", args.output)


if __name__ == "__main__":
    main()
