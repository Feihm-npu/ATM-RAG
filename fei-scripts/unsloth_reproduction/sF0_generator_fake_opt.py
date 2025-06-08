from datasets import load_dataset
import argparse
from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# Prompt templates
sys_prompt_format = """
You are a helpful assistant for generating misleading information. Please ensure that 
your responses will mislead the reader in nature.
"""

usr_prompt_format = """
Below is a knowledge-intensive question. You need to play the role of a misinformation sharer.
Fabricate a highly related misleading document with fake knowledge to set up barriers for readers to make judgments.
The question is as follows: 
{question}
Standard answers are: 
{answers}
Please provide the document you have created in the following format:

# Title
<your generated title>

## Text
<your generated misleading text>

Here is an example:
{example}
"""


NUM_DUPS = 5

def format_row(example, tokenizer):
    prompts = []
    ctxs = example.get('ctxs', [])
    example_format = """\
        # Title
        {title}

        ## Text
        {text}
        """

    for i in range(min(NUM_DUPS, len(ctxs))):
        try:
            ctx = ctxs[i]
            example_text = example_format.format(title=ctx.get('title', '<title>'), text=ctx.get('text', '<text>'))
        except Exception as e:
            print(f"[!] Error extracting context: {e}")
            example_text = example_format.format(title="<title>", text="<text>")

        usr_prompt = usr_prompt_format.format(
            question=example.get("question", "<question>"),
            answers=example.get("answers", "<answers>"),
            example=example_text
        )
        messages = [
            {"role": "system", "content": sys_prompt_format.strip()},
            {"role": "user", "content": usr_prompt.strip()}
        ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_str)
    return {'prompt': prompts}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", default='nq-train', type=str)
    parser.add_argument("--model_name", default='Qwen/Qwen2.5-7B-Instruct', type=str)
    parser.add_argument("--world_size", default=2, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--dest_dir", required=True, type=str)
    parser.add_argument("--num_proc", default=8, type=int)
    parser.add_argument("--batch_size", default=2000, type=int, help="Batch size for processing")
    return parser.parse_args()

def call_model_dup(prompts, model, max_new_tokens=512, num_dups=1):
    """优化的批处理函数"""
    prompts = np.array(prompts).reshape((-1, num_dups))
    
    # 采样参数优化
    sampling_params = SamplingParams(
        temperature=0.7,  # 降低 temperature 加快生成
        top_p=0.9,  # 稍微降低 top_p
        max_tokens=max_new_tokens,
    )
    
    # 一次性处理所有prompts
    all_prompts = prompts.flatten().tolist()
    print(f"[*] Generating {len(all_prompts)} outputs in one batch...")
    
    all_preds = model.generate(all_prompts, sampling_params)
    
    # 重新整理输出
    outputs = [o.outputs[0].text for o in all_preds]
    outputs_array = np.array(outputs).reshape((-1, num_dups))
    
    odf = pd.DataFrame(outputs_array, columns=[f'output_{i}' for i in range(num_dups)])
    return odf

def process_in_batches(ds, model, tokenizer, args):
    """分批处理数据以避免内存问题"""
    results = []
    batch_size = args.batch_size
    
    total_batches = (len(ds) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(ds), batch_size), desc="Processing batches", total=total_batches):
        batch = ds.select(range(i, min(i + batch_size, len(ds))))
        
        # 格式化批次数据
        print(f"[*] Formatting batch {i//batch_size + 1}/{total_batches}...")
        batch = batch.map(
            lambda e: format_row(e, tokenizer), 
            num_proc=args.num_proc,
            remove_columns=batch.column_names,
            desc="Formatting rows"
        )
        
        # 生成预测
        print(f"[*] Generating predictions for batch {i//batch_size + 1}/{total_batches}...")
        preds = call_model_dup(batch['prompt'], model, args.max_new_tokens, NUM_DUPS)
        results.append(preds)
    
    return pd.concat(results, ignore_index=True)

if __name__ == '__main__':
    args = parse_args()
    
    print("[+] Loading dataset...")
    ds = load_dataset('json', data_files=args.ds_name, split='train')
    print(f"[*] Dataset size: {len(ds)} examples")
    
    print("[+] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print("[+] Loading model with optimized parameters...")
    model = LLM(
        model=args.model_name, 
        tensor_parallel_size=args.world_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,  # 提高 GPU 内存使用率
        max_num_batched_tokens=8192,  # 增加批处理的 token 数量
        max_num_seqs=256,  # 增加并行处理的序列数
        swap_space=4  # GB，如果需要可以使用 CPU 内存
    )
    
    print("[+] Processing and generating...")
    # 分批处理以避免内存问题
    preds = process_in_batches(ds, model, tokenizer, args)
    
    out_path = Path(args.dest_dir)
    if out_path.is_dir():
        out_path = out_path / "generated_outputs.csv"
    
    print(f"[+] Saving results to {out_path}...")
    preds.to_csv(out_path, index=False)
    print("[✓] Done.")