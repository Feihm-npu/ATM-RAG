import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
import random
import pandas as pd
import os

# Qwen-compatible prompt template
qwen_prompt_template = """
You are a helpful assistant. Below is a question and some retrieved documents (some may be irrelevant).
Use them to write a high-quality, concise, and accurate answer.

[Knowledge]
{paragraph}

Question: {question}

Answer:
"""

# Markdown paragraph formatting template
paragraph_template = """
# Title: {title}
## text: {text}
"""

def split_ctxs(example):
    ctxs_true = []
    ctxs_false = []
    for ctx in example['ctxs']:
        if ctx.get('hasanswer', False):
            ctxs_true.append(ctx)
        else:
            ctxs_false.append(ctx)
    hasanswer = len(ctxs_true) > 0
    return {
        'question': example['question'],
        'answers': example['answers'],
        'ctxs_true': ctxs_true,
        'ctxs_false': ctxs_false,
        'hasanswer': hasanswer,
    }

def process_data(example):
    if not example['hasanswer']:
        return None

    ctxs_true = example['ctxs_true']
    ctxs_false = example['ctxs_false']

    random.shuffle(ctxs_true)
    random.shuffle(ctxs_false)

    combined_ctxs = ctxs_true + ctxs_false
    combined_ctxs = combined_ctxs[:10]

    formatted_docs = [
        f"[document] {paragraph_template.format(title=ctx['title'], text=ctx['text']).strip()} [/document]"
        for ctx in combined_ctxs
    ]
    raw_paragraph = "\n".join(formatted_docs)

    return {
        "paragraph": raw_paragraph,
        "question": example['question'],
        "answer": example['answers'][0] if example['answers'] else '',
    }

def map_to_qwen_src_tgt(example):
    source = qwen_prompt_template.format(
        paragraph=example['paragraph'],
        question=example['question']
    )
    target = example['answer']
    return {"source": source, "target": target}

def process_str_to_input_ids(example, tokenizer):
    src_ids = tokenizer.encode(example['source'], add_special_tokens=False)
    tgt_ids = tokenizer.encode(example['target'], add_special_tokens=False)

    input_ids = src_ids + tgt_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(src_ids) + tgt_ids + [tokenizer.eos_token_id]

    input_ids = input_ids[:tokenizer.model_max_length - 1]
    labels = labels[:tokenizer.model_max_length - 1]

    return {
        "input_ids": input_ids,
        "labels": labels
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dst_path', type=str, required=True)
    parser.add_argument('--csv_preview_path', type=str, default=None, help='Optional path to save 50-sample CSV preview')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ds = load_dataset('json', data_files=args.data_path, split='train')

    ds = ds.map(split_ctxs, remove_columns=ds.column_names, num_proc=8)
    ds = ds.filter(lambda x: x['hasanswer'], num_proc=8)
    ds = ds.map(process_data, remove_columns=ds.column_names, num_proc=8)
    ds = ds.map(map_to_qwen_src_tgt, remove_columns=ds.column_names, num_proc=8)

    # 保存前50条为 CSV（可选）
    if args.csv_preview_path is None:
        args.csv_preview_path = os.path.join(args.dst_path,'view.csv')

    if args.csv_preview_path:
        df = ds.select(range(min(50, len(ds)))).to_pandas()
        df[['source', 'target']].to_csv(args.csv_preview_path, index=False)
        print(f"[✓] Preview CSV saved to: {args.csv_preview_path}")

    ds = ds.map(lambda x: process_str_to_input_ids(x, tokenizer), remove_columns=ds.column_names, num_proc=8)
    ds.save_to_disk(args.dst_path)
    print(f"[✓] Processed Qwen-format S2S data saved to: {args.dst_path}")

if __name__ == '__main__':
    main()