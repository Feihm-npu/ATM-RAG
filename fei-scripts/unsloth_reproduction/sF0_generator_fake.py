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
    return parser.parse_args()

def call_model_dup(prompts, model, max_new_tokens=512, num_dups=1):
    prompts = np.array(prompts)
    prompts = prompts.reshape((-1, num_dups))
    pdf = pd.DataFrame(prompts, columns=[f'input_{i}' for i in range(num_dups)])

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)
    odf = pd.DataFrame(columns=[f'output_{i}' for i in range(num_dups)])

    for i in tqdm(range(num_dups), desc="Generating outputs"):
        preds = model.generate(pdf[f'input_{i}'].tolist(), sampling_params)
        odf[f'output_{i}'] = [o.outputs[0].text for o in preds]
    return odf

if __name__ == '__main__':
    args = parse_args()
    print("[+] Loading dataset...")
    # ds = load_dataset('json', data_files=args.ds_name, split='train').select(range(100))
    ds = load_dataset('json', data_files=args.ds_name, split='train')
    print("[+] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("[+] Formatting dataset...")
    ds = ds.map(lambda e: format_row(e, tokenizer), num_proc=args.num_proc, remove_columns=ds.column_names, desc="Formatting rows")

    print("[+] Loading model...")
    model = LLM(model=args.model_name, tensor_parallel_size=args.world_size, trust_remote_code=True, disable_custom_all_reduce=True)

    print("[+] Generating predictions...")
    preds = call_model_dup(ds['prompt'], model, max_new_tokens=args.max_new_tokens, num_dups=NUM_DUPS)

    out_path = Path(args.dest_dir)
    if out_path.is_dir():
        out_path = out_path / "generated_outputs.csv"

    print(f"[+] Saving results to {out_path}...")
    preds.to_csv(out_path, index=False)
    print("[✓] Done.")