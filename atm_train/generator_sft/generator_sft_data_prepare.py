#!/usr/bin/env python
# coding: utf-8

# 导入必要的库
from datasets import load_dataset  # 用于加载和处理数据集
import numpy as np
from shuffle import Shuffler  # 自定义的文档打乱工具
import random
from transformers import AutoTokenizer  # 用于文本标记化
from pathlib import Path
import argparse



# 定义不同任务的提示模板
prompt_template = {
    "deshuffle": "atm_deshuffle",  # 用于重新排序打乱的文档
    "gt_doc": "atm_gt_doc",  # 用于处理黄金文档
    "rag_qa": "atm_instruct",  # 用于RAG问答
    "close_qa": "atm_instruct_close",  # 用于封闭式问答
    "cot_deshuffle_qa": [  # 思维链+重排序+问答
        "atm_deshuffle",
        "atm_cot_qa_suffix"
    ],
    "cot_gt_doc_qa": [  # 思维链+黄金文档+问答
        "atm_gt_doc",
        "atm_cot_qa_suffix"
    ]
}

# 定义打乱配置，控制不同级别文本单元的打乱程度
shuffle_config = {
    "paragraph": {
        "shuffle_degree": 0.,  # 打乱程度
        "drop_ratio": 0,       # 丢弃比例
        "duplicate_ratio": 0.  # 复制比例
    },
    "passage": {
        "shuffle_degree": 0., 
        "drop_ratio": 0., 
        "duplicate_ratio": 0.
    },
    "sentence": {
        "shuffle_degree": 0., 
        "drop_ratio": 0., 
        "duplicate_ratio": 0.
    }
}

shuffler = Shuffler(shuffle_config) 

from prompting_for_rag import get_prompt_template, get_prompt  # 导入提示模板

# 定义示例格式，用于构建提示
example_format = 'TITLE {title} # TEXT {text}'

def process_data(example):
    """
    处理原始数据，提取相关字段并应用打乱操作
    
    参数:
        example: 数据集中的一个样本
    返回:
        处理后的样本字典
    """
    # 从ctxs列表中提取字段
    dict_of_lists = {key: [d[key] for d in example["ctxs"]] for key in example["ctxs"][0]}
    example["passages"] = {}
    example["passages"]["is_selected"] = dict_of_lists["hasanswer"]  # 标记哪些段落包含答案
    example["passages"]["passage_text"] = dict_of_lists["text"]      # 段落文本
    raw_psgs = example['passages']['passage_text'][:10]  # 取前10个段落
    selected = example['passages']['is_selected'][:10]   # 对应的标记

    assert len(raw_psgs) == len(selected)
    
    # 确保 selected 中没有 None 值
    if None in selected:
        selected = [False if x is None else x for x in selected]
    
    psgs = raw_psgs.copy()
    new_selected = selected.copy()
    
    # 以100%的概率打乱段落顺序
    if random.uniform(0, 1) < 1 and psgs is not None:
        psgs, new_selected = shuffler.shuffle_passage_list(psgs, new_selected)

    question = example['question']

    # 获取黄金文档(包含答案的文档)
    gt_doc = raw_psgs[np.argmax(selected)] if any(selected) else "None"

    raw_passages = "None"
    tar_passages = "None"

    # 格式化原始段落
    raw_evidences = ["[document] {} [/document]".format(ctx) for ctx in psgs]
    raw_passages = "\n".join(raw_evidences)
    
    # 构建目标段落集合(包含黄金文档和部分负样本)
    if any(selected):
        tar_psgs = [raw_psgs[np.argmax(selected)], ]  # 首先添加黄金文档

        # 以85%的概率排除非答案文档
        for idx, one_psg in enumerate(raw_psgs):
            if random.uniform(0, 1) > 0.15 and selected[idx] == 0:
                tar_psgs.append(one_psg)
    
        tar_evidences = ["[document] {} [/document]".format(ctx) for ctx in tar_psgs]
        tar_passages = "\n".join(tar_evidences)

    # 返回处理后的样本
    return {
        "paragraph": raw_passages,  # 原始段落
        "question": question,       # 问题
        "doc_list": tar_passages,   # 目标段落列表
        "gt_doc": gt_doc,           # 黄金文档
        "answer": example['answers'][0],  # 答案
        "gt_pos": np.argmax(new_selected) / len(new_selected) if any(new_selected) else 0.  # 黄金文档的相对位置
    }


def map_to_src_tgt(example):
    """
    将处理后的样本映射为源文本和目标文本对
    
    参数:
        example: 处理后的样本
    返回:
        包含source和target的字典
    """
    rnd = random.uniform(0, 1)
    mode = None
    
    # 当前实现: 30%概率为close_qa任务，70%概率为rag_qa任务
    if rnd < 0.3:
        mode = "close_qa"
    else:
        mode = "rag_qa"

    # 如果模式是top1_qa，使用黄金文档作为段落，并转为rag_qa模式
    if mode == 'top1_qa':
        example['paragraph'] = example['gt_doc']
        mode = 'rag_qa'
    
    # 根据不同模式构建源文本和目标文本
    if mode == "deshuffle":
        return {
            "source": [prompt_template[mode].format_map(example)],
            "target": [example["doc_list"]]
        }
    elif mode in ("rag_qa", "close_qa"):
        return {
            "source": [prompt_template[mode].format_map(example)],
            "target": [example["answer"]]
        }
    elif mode == "gt_doc":
        return {
            "source": [prompt_template[mode].format_map(example)],
            "target": [example["gt_doc"]]
        }
    else:
        # 处理多步骤提示(如思维链)
        srcs = []; tgts = []
        for sent in prompt_template[mode]:
            srcs.append(sent.format_map(example))
        if mode == "cot_deshuffle_qa":
            tgts = [example["doc_list"], example["answer"]]
        elif mode == "cot_gt_doc_qa":
            tgts = [example["gt_doc"], example["answer"]]

        return {
            "source": srcs,
            "target": tgts
        }


def process_str_to_input_ids(example, tokenizer):
    """
    将文本转换为模型输入的token ID
    
    参数:
        example: 包含source和target的样本
        tokenizer: 用于文本编码的tokenizer
    返回:
        包含input_ids和labels的字典
    """
    input_ids = []; labels = []
    
    # 处理每对源文本和目标文本
    for one_src, one_tgt in zip(example['source'], example['target']):
        # 编码源文本，并将其标签设为-100(在计算损失时会被忽略)
        src_ids = tokenizer.encode(one_src)
        src_labels = [-100] * len(src_ids)
        
        # 编码目标文本，保留其标签用于训练
        tgt_ids = tokenizer.encode(one_tgt, add_special_tokens=False)
        tgt_labels = tgt_ids.copy()

        # 合并源文本和目标文本的token ID和标签
        input_ids += src_ids + tgt_ids
        labels += src_labels + tgt_labels

        # 添加结束标记
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    # 截断到模型最大长度
    input_ids = input_ids[:tokenizer.model_max_length - 1]
    labels = labels[:tokenizer.model_max_length - 1]
    
    return {
        "input_ids": input_ids,
        "labels": labels
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Process dataset for RAG tasks')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--dst_path', type=str, required=True, help='Destination path to save processed data')
    
    args = parser.parse_args()

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # 初始化文档打乱器
    shuffler = Shuffler(shuffle_config)    

    # 加载数据集
    ds = load_dataset('json', data_files=args.data_path, split='train') 

    # 数据处理流程
    # 1. 处理原始数据
    ds = ds.map(process_data, remove_columns=ds.column_names, num_proc=8)

    # 2. 映射为源文本和目标文本
    ds = ds.map(map_to_src_tgt, remove_columns=ds.column_names, num_proc=8)

    # 3. 转换为token ID，需要传递tokenizer
    ds = ds.map(lambda x: process_str_to_input_ids(x, tokenizer), remove_columns=ds.column_names, num_proc=8, desc="Converting to input IDs")

    # 保存处理后的数据集
    ds.save_to_disk(args.dst_path)

    print(f"[✓] Processed dataset saved to: {args.dst_path}")

if __name__ == "__main__":
    main()