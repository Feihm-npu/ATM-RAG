import argparse
import os
import numpy as np
import torch
import wandb
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
from datasets import load_dataset
from transformers import TrainingArguments
from trl import DPOTrainer

# -------- Tokenization Helpers --------
truncation_mode = 'keep_end'
label_pad_token_id = -100
max_prompt_length = 3072
max_length = 4096

def build_tokenized_answer(prompt, answer, tokenizer):
    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    response_token_ids_start_idx = len(prompt_input_ids)
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )

def tokenize_row(feature, tokenizer):
    batch = {}
    prompt = feature["prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

    chosen_tokens = build_tokenized_answer(prompt, chosen, tokenizer)
    rejected_tokens = build_tokenized_answer(prompt, rejected, tokenizer)

    prompt_len_input_ids = min(len(chosen_tokens["prompt_input_ids"]), len(rejected_tokens["prompt_input_ids"]))
    for k, v in prompt_tokens.items():
        prompt_tokens[k] = v[:prompt_len_input_ids]

    if tokenizer.bos_token_id is not None:
        for t in [prompt_tokens, chosen_tokens, rejected_tokens]:
            t["prompt_input_ids"] = [tokenizer.bos_token_id] + t["prompt_input_ids"]
            t["prompt_attention_mask"] = [1] + t["prompt_attention_mask"]

    for t in [chosen_tokens, rejected_tokens]:
        t["input_ids"].append(tokenizer.eos_token_id)
        t["attention_mask"].append(1)

    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))
    for t in [chosen_tokens, rejected_tokens, prompt_tokens]:
        if len(t["prompt_input_ids"]) + longer_response_length > max_length:
            if truncation_mode == "keep_end":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    t[k] = t[k][-max_prompt_length:]

    for t in [chosen_tokens, rejected_tokens]:
        if len(t["prompt_input_ids"]) + longer_response_length > max_length:
            for k in ["input_ids", "attention_mask"]:
                t[k] = t[k][:max_length - max_prompt_length]

    def make_labels(t):
        labels = t["prompt_input_ids"] + t["input_ids"]
        labels[:len(t["prompt_input_ids"])] = [label_pad_token_id] * len(t["prompt_input_ids"])
        return labels

    for prefix, toks in zip(["chosen_", "rejected_", ""], [
        {**chosen_tokens, "labels": make_labels(chosen_tokens)},
        {**rejected_tokens, "labels": make_labels(rejected_tokens)},
        prompt_tokens
    ]):
        for k, v in toks.items():
            if k != "token_type_ids":
                batch[f"{prefix}{k}"] = v

    return batch

# -------- Main Training Script --------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-7B")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./s2_experiments/model_1epoch_dpo")
    parser.add_argument("--wandb_project", type=str, default="unsloth-dpo")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    wandb.init(project=args.wandb_project)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_length,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw_dataset = load_dataset("json", data_files=f'{args.train_file}', split="train")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        bf16=args.bf16,
        logging_steps=1,
        report_to="wandb",
        run_name=f"dpo-{os.path.basename(args.output_dir)}",
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        save_strategy="no",
        dataloader_num_workers=1,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=0.2,
        train_dataset=raw_dataset,
        eval_dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        processing_class=tokenize_row,
        max_prompt_length=max_prompt_length,
        dataset_num_proc=8,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()