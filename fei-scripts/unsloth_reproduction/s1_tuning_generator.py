import unsloth
import os
import argparse
import random
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, set_seed, Seq2SeqTrainer, Seq2SeqTrainingArguments
from unsloth import FastLanguageModel

import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="unsloth-clm")
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

    wandb.init(project=args.wandb_project)

    # Load model and tokenizer via Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=4096,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=args.seed,
        use_rslora=False,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load dataset from disk
    try:
        dataset = DatasetDict.load_from_disk(args.train_data)['train']
    except:
        dataset = load_from_disk(args.train_data)
    # print("Dataset columns:", dataset.column_names)
    # print("Example:", dataset[0])


    # Training arguments
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
        report_to="wandb",
        run_name=f"unsloth-{os.path.basename(args.model_name_or_path)}",
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
    # trainer.save_model(os.path.join(args.output_dir, "model_final"))
    model.save_pretrained(os.path.join(args.output_dir, "model_final"))

# 保存 tokenizer（否则下一次 tokenizer 也加载不了）
    tokenizer.save_pretrained(os.path.join(args.output_dir, "model_final"))
    wandb.finish()

if __name__ == "__main__":
    main()
