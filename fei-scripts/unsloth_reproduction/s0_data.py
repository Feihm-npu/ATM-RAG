import os
import unsloth
import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from prompting_for_rag import get_prompt
from unsloth import FastLanguageModel
from transformers import GenerationConfig
from transformers import TextStreamer

NUM_DUPS = 5
example_format = 'TITLE {title} # TEXT {text}'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", default="nq-train", type=str)
    parser.add_argument("--model_dir", type=str, required=True, help="Path to s2.py 训练得到的模型")
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--bz", default=2, type=int)
    parser.add_argument("--dest_dir", required=True, type=str)
    parser.add_argument("--num_dups", type=int, default=5)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()

def format_row(example):
    prompts = []
    for i in range(NUM_DUPS):
        try:
            passage = example['passages']['passage_text'][i]
        except:
            try:
                passage = example['passages']['passage_text'][0]
            except:
                passage = {"title": "<title>", "text": "<text>"}
        item = {
            "question": example['question'],
            "answers": example['answers'],
            "example": example_format.format_map(passage)
        }
        prompts.append(get_prompt("atm_data_attacker", item))
    return {"prompt": prompts}

from tqdm import tqdm  # 确保你已安装 tqdm

def call_model_up(prompts, model, tokenizer, max_new_tokens=50, num_dups=5, batch_size=500):
    prompts = np.array(prompts)
    prompts = prompts.reshape((-1, num_dups))
    pdf = pd.DataFrame(prompts, columns=[f'input_{idx}' for idx in range(num_dups)])
    odf = pd.DataFrame(columns=[f'output_{idx}' for idx in range(num_dups)])

    # tqdm 外层进度条：对每个重复轮次
    for idx in range(num_dups):
        input_texts = pdf[f'input_{idx}'].tolist()
        outputs = []

        for i in tqdm(range(0, len(input_texts), batch_size), desc=f"Generating output_{idx}"):
            batch_inputs = input_texts[i:i+batch_size]
            encoded = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=4096)
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

            with torch.no_grad():
                batch_outputs = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            # 删除 prompt 部分，只保留回答
            cleaned = [
                text[len(batch_inputs[j]):].strip() if text.startswith(batch_inputs[j]) else text.strip()
                for j, text in enumerate(decoded)
            ]
            outputs.extend(cleaned)

        odf[f'output_{idx}'] = outputs

    return pdf.join(odf)





def main():
    args = parse_args()
    global NUM_DUPS
    NUM_DUPS = args.num_dups

    dataset = load_dataset("json", data_files=f"/home/feihm/llm-fei/Data/{args.ds_name}.jsonl", split="train")
    dataset = dataset.map(format_row, num_proc=8, remove_columns=dataset.column_names)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B",  # base model
        max_seq_length=4096,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        load_in_4bit=False
    )
    model.load_adapter(args.model_dir)

    FastLanguageModel.for_inference(model)  # ✅ 启用推理加速 + 显存节省
    # text_streamer = TextStreamer(tokenizer)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    preds = call_model_up(dataset["prompt"], model, tokenizer, max_new_tokens=args.max_new_tokens, num_dups=args.num_dups, batch_size=args.bz)
    preds.to_csv(dest_dir / f"{args.ds_name}_fab.csv", index=False)

    # result_df = generate_multiple(model, tokenizer, dataset['prompt'], args.num_dups, args.max_new_tokens)
    # result_df.to_csv(dest_dir / f"{args.ds_name}_fab.csv", index=False)
    print(f"[\u2713] Finished generating to {dest_dir}/{args.ds_name}_fab.csv")

if __name__ == "__main__":
    main()