import os
import argparse
import random
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    set_seed, AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer,Seq2SeqTrainingArguments,DataCollatorForSeq2Seq
)
import wandb
from accelerate import Accelerator
accelerator = Accelerator()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="unsloth-clm")
    parser.add_argument("--deepspeed", type=str, default="ds_configs/ds_config_zero2.json")
    return parser.parse_args()

def seed_everything(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    seed_everything(args.seed)

    # Only main process initializes wandb
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=f"Qwen-Finetune", reinit=True)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        dataset = DatasetDict.load_from_disk(args.train_data)['train']
    except:
        dataset = load_from_disk(args.train_data)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=1,
        bf16=args.bf16,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        report_to="wandb",
        run_name=f"Qwen-{os.path.basename(args.model_name_or_path)}",
        save_strategy="no",
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        tokenizer=tokenizer,
    )

    trainer.train()

    if accelerator.is_main_process:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        wandb.finish()

    # Ensure clean exit of NCCL
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
