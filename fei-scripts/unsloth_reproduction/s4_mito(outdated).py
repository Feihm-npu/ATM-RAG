import os
import unsloth
from unsloth import FastLanguageModel, PatchDPOTrainer
# PatchDPOTrainer()
import argparse
import random
import numpy as np
import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, set_seed, AutoModelForCausalLM

from mito import MITOTrainer, mito_tokenize_row, MITODataCollatorWithPadding  # 自定义模块

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_mito")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="unsloth-mito")
    parser.add_argument("--beta", type=float, default=0.1)  # mito 的 KL loss 权重
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
    model.train()

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
    )
    ref_model.eval()  # 冻结参考模型
    for param in ref_model.parameters():
        param.requires_grad = False

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 加载原始 JSON 数据集
    # raw_datasets = load_dataset("json", data_files=args.train_data, split="train")
    # tokenized_dataset = raw_datasets.select(range(200)).map(
    #     lambda x: mito_tokenize_row(x, tokenizer),
    #     # remove_columns=raw_datasets.column_names,
    #     desc="Tokenizing...",
    # )
    dummy_data = [
        {
            "prompt": "Tell me a story about a brave knight.",
            "adv_prompt": "Write a shopping list for vegetables.",
            "answer": "Once upon a time, a brave knight set out to rescue the princess from a dragon."
        },
        {
            "prompt": "What is the capital of Italy?",
            "adv_prompt": "What is the color of the sky?",
            "answer": "Rome"
        },
        {
            "prompt": "Explain the process of photosynthesis.",
            "adv_prompt": "How do you cook pasta?",
            "answer": "Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water."
        }
    ] * 10



    dataset = Dataset.from_list(dummy_data)
    tokenized_dataset = dataset.map(lambda x: mito_tokenize_row(x, tokenizer))

    tokenized_dataset = tokenized_dataset.add_column("chosen", [""] * len(tokenized_dataset))
    tokenized_dataset = tokenized_dataset.add_column("rejected", [""] * len(tokenized_dataset))

    for i in range(2):
        print(f"Sample {i}:")
        print("chosen_input_ids:", tokenized_dataset[i]["chosen_input_ids"])
        print("chosen_labels:", tokenized_dataset[i]["chosen_labels"])
        print("rejected_labels:", tokenized_dataset[i]["rejected_labels"])
        print("→ #labels!=-100:",
            sum(1 for x in tokenized_dataset[i]["chosen_labels"] if x != -100),
            "| length:", len(tokenized_dataset[i]["chosen_labels"]))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=1,
        bf16=args.bf16,
        report_to="wandb",
        run_name=f"mito-{os.path.basename(args.model_name_or_path)}",
        save_strategy="no",
        remove_unused_columns=False,
    )

    trainer = MITOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=MITODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=-100,
            is_encoder_decoder=False
        ),
        beta=0.1,
        loss_type="sigmoid",
        # preprocessed=True,
        # auto_prepare_dataset=False,   # ← 关键参数 1
        # auto_tokenize_dataset=False,  # ← 关键参数 2
    )


    trainer.train()

    trainer.save_model(os.path.join(args.output_dir, "model_mito_final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "model_mito_final"))
    wandb.finish()

if __name__ == "__main__":
    main()
