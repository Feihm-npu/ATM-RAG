import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer


NUM_DUPS = 1

def format_row(example, tokenizer):
    ctxs = example['ctxs']
    question = example['question']

    paragraph = "\n\n".join(
        f"{ctx.get('title', '')}. {ctx.get('text', '')}" for ctx in ctxs
    )

    prompt = """Based on the following document, answer the question with a very short and accurate phrase.
    \n\nDocument:\n{paragraph}\n\nQuestion:\n{question}"""

    messages = [
        {"role": "system", "content": "You are a helpful, concise, and knowledgeable assistant."},
        {"role": "user", "content": prompt.format(paragraph=paragraph, question=question)}
    ]

    # 使用模型的 chat 模板（自动处理 <|im_start|> 等标记）
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return {'prompt': chat_prompt}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", default='nq-train', type=str)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--world_size", default=2, type=int)
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--dest_dir", required=True, type=str)
    return parser.parse_args()

def call_model_dup(prompts, model, max_new_tokens=256, num_dups=1):
    prompts = np.array(prompts).reshape((-1, num_dups))
    pdf = pd.DataFrame(prompts, columns=[f'input_{idx}' for idx in range(num_dups)])

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens
    )

    odf = pd.DataFrame(columns=[f'output_{idx}' for idx in range(num_dups)])

    for idx in tqdm(range(num_dups), desc="Generating outputs"):
        raw_outputs = model.generate(pdf[f'input_{idx}'].tolist(), sampling_params)
        preds = [out.outputs[0].text.strip() for out in raw_outputs]
        odf[f'output_{idx}'] = preds

    return odf

if __name__ == '__main__':
    args = parse_args()
    print(">> Loading dataset...")
    ds = load_dataset('json', data_files=args.ds_name, split='train', num_proc=8)
    ds = ds.select(range(1000))
    print(">> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(">> Formatting dataset...")
    ds = ds.map(lambda x: format_row(x, tokenizer), num_proc=8, remove_columns=ds.column_names)


    print(">> Loading model...")
    model = LLM(model=args.model_name, tensor_parallel_size=args.world_size)

    print(">> Generating predictions...")
    preds = call_model_dup(ds['prompt'], model, max_new_tokens=args.max_new_tokens, num_dups=NUM_DUPS)

    print(">> Saving results...")
    preds.to_csv(args.dest_dir, index=False)
    print(f"[✓] Results saved to: {args.dest_dir}")
