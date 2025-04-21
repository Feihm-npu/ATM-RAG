import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

# 为 instruct 模型准备的 prompt 模板（使用 ChatML 风格）
PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful, concise, and knowledgeable assistant.
<|im_end|>
<|im_start|>user
Based on the following document, answer the question with a very short and accurate phrase.

Document:
{paragraph}

Question:
{question}
<|im_end|>
<|im_start|>assistant
"""

NUM_DUPS = 5

def format_row(example):
    prompts = []
    ctxs = example['ctxs']
    question = example['question']

    for i in range(min(NUM_DUPS, len(ctxs))):
        ctx = ctxs[i]
        paragraph = f"{ctx.get('title', '')}. {ctx.get('text', '')}"
        prompt = PROMPT_TEMPLATE.format(paragraph=paragraph, question=question)
        prompts.append(prompt)

    while len(prompts) < NUM_DUPS:
        paragraph = "No document available."
        prompt = PROMPT_TEMPLATE.format(paragraph=paragraph, question=question)
        prompts.append(prompt)

    return {'prompt': prompts}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", default='nq-train', type=str)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--world_size", default=2, type=int)
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--dest_dir", required=True, type=str)
    return parser.parse_args()

def call_model_dup(prompts, model, max_new_tokens=32, num_dups=1):
    prompts = np.array(prompts).reshape((-1, num_dups))
    pdf = pd.DataFrame(prompts, columns=[f'input_{idx}' for idx in range(num_dups)])

    sampling_params = SamplingParams(
        temperature=0.0,  # 使用 greedy 解码以获得 deterministic 输出
        top_p=1.0,
        max_tokens=max_new_tokens
    )

    odf = pd.DataFrame(columns=[f'output_{idx}' for idx in range(num_dups)])

    for idx in tqdm(range(num_dups), desc="Generating outputs"):
        raw_outputs = model.generate(pdf[f'input_{idx}'].tolist(), sampling_params)
        # 截取第一个输出并 strip 空格
        preds = [out.outputs[0].text.strip() for out in raw_outputs]
        odf[f'output_{idx}'] = preds

    return odf

if __name__ == '__main__':
    args = parse_args()
    print(">> Loading dataset...")
    ds = load_dataset('json', data_files=args.ds_name, split='train')

    print(">> Formatting dataset...")
    ds = ds.map(format_row, num_proc=8, remove_columns=ds.column_names)

    print(">> Loading model...")
    model = LLM(model=args.model_name, tensor_parallel_size=args.world_size, trust_remote_code=True, disable_custom_all_reduce=True)

    print(">> Generating predictions...")
    preds = call_model_dup(ds['prompt'], model, max_new_tokens=args.max_new_tokens, num_dups=NUM_DUPS)

    print(">> Saving results...")
    preds.to_csv(args.dest_dir, index=False)
    print(f"[✓] Results saved to: {args.dest_dir}")
