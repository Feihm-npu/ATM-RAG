import argparse
import os
import numpy as np
import torch
import wandb
from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from accelerate import Accelerator

# -------- Tokenization Config --------
truncation_mode = 'keep_end'
label_pad_token_id = -100
max_prompt_length = 3072
max_length = 4096

# -------- Tokenization Functions --------
def build_tokenized_answer(prompt, answer, tokenizer):
    full_tokenized = tokenizer(prompt + answer)
    prompt_input_ids = tokenizer(prompt)["input_ids"]

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



# -------- Custom DPO --------
class ATM_DPOTrainer(DPOTrainer):
    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        batch = {}
        prompt = features["prompt"]
        chosen = features["chosen"]
        rejected = features["rejected"]

        prompt_tokens = processing_class(prompt)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        chosen_tokens = build_tokenized_answer(prompt, chosen, processing_class)
        rejected_tokens = build_tokenized_answer(prompt, rejected, processing_class)

        prompt_len_input_ids = min(len(chosen_tokens["prompt_input_ids"]), len(rejected_tokens["prompt_input_ids"]))
        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        if processing_class.bos_token_id is not None:
            for t in [prompt_tokens, chosen_tokens, rejected_tokens]:
                t["prompt_input_ids"] = [processing_class.bos_token_id] + t["prompt_input_ids"]
                t["prompt_attention_mask"] = [1] + t["prompt_attention_mask"]

        for t in [chosen_tokens, rejected_tokens]:
            t["input_ids"].append(processing_class.eos_token_id)
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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ref_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./s2_experiments/model_1epoch_dpo")
    parser.add_argument("--wandb_project", type=str, default="dpo")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project)

    model = args.model_name
    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    ref_model = args.ref_model
    ref_model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw_dataset = load_dataset("json", data_files=f'{args.train_file}', split="train")
    # preprocessor = CustomDPOPreprocessor(tokenizer)
    # with accelerator.main_process_first():
    #     tokenized_dataset = raw_dataset.map(
    #         lambda x: preprocessor([x])[0],
    #         remove_columns=raw_dataset.column_names,
    #         num_proc=8,
    #         desc="Tokenizing with custom preprocessor"
    #     )
    #     print(f'Processed dataset: {tokenized_dataset}')


    training_args = DPOConfig(
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
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
        dataloader_num_workers=4,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        dataset_num_proc=32,
    )

    

    trainer = ATM_DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        # beta=0.2,
        processing_class=tokenizer,
        # processing_class=tokenize_row,
        train_dataset=raw_dataset,
        eval_dataset=raw_dataset,
    )

    trainer.train()

    if accelerator.is_main_process:
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        wandb.finish()

if __name__ == "__main__":
    main()
