from prompting_for_rag import get_prompt
from datasets import load_dataset, Dataset
import json
import argparse
from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 定义示例格式，用于构建提示
example_format = 'TITLE {title} # TEXT {text}'

# 每个问题生成的重复次数
NUM_DUPS = 5

def format_row(example):
    """
    将数据集中的每一行转换为提示格式
    参数:
        example: 数据集中的一行
    返回:
        包含多个提示的字典
    """
    prompts = []
    ctxs = example['ctxs']  # 获取上下文列表

    for i in range(min(NUM_DUPS, len(ctxs))):  # 确保不超出ctxs的长度
        item = {}
        try:
            # 从ctxs列表中的当前元素提取title和text
            ctx = ctxs[i]
            item['example'] = example_format.format_map({
                "title": ctx['title'],
                "text": ctx['text']
            })
        except (KeyError, IndexError, AttributeError):
            # 如果提取失败，使用占位符
            item['example'] = example_format.format_map({
                "title": "<title>",
                "text": "<text>",
            })

        # 添加问题和答案
        item['question'] = example['question']
        item['answers'] = example['answers']
        # 使用预定义的提示模板生成完整提示
        prompts.append(get_prompt('atm_data_attacker', item))
    
    # 如果ctxs长度不足NUM_DUPS，用占位符补齐
    while len(prompts) < NUM_DUPS:
        item = {
            'example': example_format.format_map({
                "title": "<title>",
                "text": "<text>",
            }),
            'question': example['question'],
            'answers': example['answers']
        }
        prompts.append(get_prompt('atm_data_attacker', item))

    return {'prompt': prompts}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--ds_name", default='nq-train', type=str)  # 数据集名称
    parser.add_argument("--model_name", default='/home/feihm/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/', type=str)  # 模型路径
    parser.add_argument("--world_size", default=2, type=int)  # 并行大小
    parser.add_argument("--max_new_tokens", default=512, type=int)  # 生成的最大token数
    parser.add_argument("--dest_dir", required=True, type=str)  # 输出目录
    
    args = parser.parse_args()
    return args

def call_model_dup(prompts, model, max_new_tokens=50, num_dups=1):
    """
    调用模型生成多个回复，并显示进度条
    参数:
        prompts: 提示列表
        model: 加载的模型
        max_new_tokens: 生成的最大token数
        num_dups: 每个提示的重复次数
    返回:
        包含模型输出的DataFrame
    """
    # 将提示重塑为二维数组，每行包含num_dups个重复
    prompts = np.array(prompts)
    prompts = prompts.reshape((-1, num_dups))
    pdf = pd.DataFrame(prompts, columns=[f'input_{idx}' for idx in range(num_dups)])
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)
    
    # 创建输出DataFrame
    odf = pd.DataFrame(columns=[f'output_{idx}' for idx in range(num_dups)])
    
    # 为每个输入列添加进度条
    for idx in tqdm(range(num_dups), desc="Generating outputs"):
        # 对每个重复的提示调用模型
        preds = model.generate(pdf[f'input_{idx}'].tolist(), sampling_params)
        preds = [pred.outputs[0].text for pred in preds]
        odf[f'output_{idx}'] = preds
    
    return odf

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    ds_name = args.ds_name
    dest_dir = Path(args.dest_dir)
    print("Loading dataset...")
    # 加载数据集
    ds = load_dataset('json', data_files=f'{ds_name}', split='train')
    
    print("Formatting dataset...")
    # 将数据集转换为提示格式，并添加进度条
    ds = ds.map(format_row, num_proc=8, remove_columns=ds.column_names, desc="Formatting rows")
    
    print("Loading model...")
    # 加载模型
    model = LLM(model=args.model_name, tensor_parallel_size=args.world_size, trust_remote_code=True, disable_custom_all_reduce=True)
    print("Generating predictions...")
    # 调用模型生成回复
    preds = call_model_dup(ds['prompt'], model, max_new_tokens=args.max_new_tokens, num_dups=NUM_DUPS)

    print("Saving results...")
    # 创建输出目录
    dest_dir = Path(args.dest_dir)
    if not dest_dir.exists():
        dest_dir.mkdir()

    model_name = Path(args.model_name).name

    # 保存结果到CSV文件
    preds.to_csv(f'{dest_dir}', index=False)
    print(f"[✓] Results saved to: {dest_dir}")