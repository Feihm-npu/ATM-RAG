import argparse
import json
from datasets import load_dataset
import random

def build_mito_dataset(source_path, dest_path):
    # 加载数据集
    try:
        data = load_dataset("json", data_files=source_path, split="train")
    except Exception as e:
        print(f"Error loading dataset from {source_path}: {e}")
        exit(1)

    # 新数据集列表
    new_dataset = []

    # 遍历原始数据集
    for sample in data:
        question = sample['question']
        answers = sample['answers']
        ctxs = sample['ctxs']

        # 筛选符合条件的段落
        chosen_paragraphs = []  # id 不以 "fab" 开头 且 hasanswer 为 True
        rejected_paragraphs = []  # 所有 "fab" 开头的段落 + 部分 id 不以 "fab" 开头 但 hasanswer 为 True 的段落

        # 分类段落
        for ctx in ctxs:
            paragraph = f"Title: {ctx['title']}\nText: {ctx['text']}"
            if ctx['id'].startswith('fab'):
                # 所有 "fab" 开头的段落都放入 rejected
                rejected_paragraphs.append(paragraph)
            else:
                if ctx['hasanswer']:
                    # id 不以 "fab" 开头 且包含答案的段落
                    chosen_paragraphs.append(paragraph)

        # 如果 chosen_paragraphs 为空，跳过此样本
        if not chosen_paragraphs:
            continue

        # 从 chosen_paragraphs 中随机选择一部分放入 rejected（假设选择 50%）
        random.shuffle(chosen_paragraphs)
        split_idx = max(1, len(chosen_paragraphs) // 2)  # 至少保留一个在 chosen
        rejected_from_chosen = chosen_paragraphs[split_idx:]
        chosen_final = chosen_paragraphs[:split_idx]
        rejected_paragraphs.extend(rejected_from_chosen)

        # 构造 prompt（使用第一个 chosen 段落作为示例）
        first_chosen = chosen_final[0]
        prompt = f"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
Always answer as helpfully as possible,
Please ensure that your responses are concise, informative and accurate . 
Write a high-quality answer for the given question using the provided search results as external knowledge 
(some of which might be irrelevant).<</SYS>>
Knowledge : 
{first_chosen} ##
Could you please help me to find the document that can help me give a correct answer to the question ?
Question: {question} ##
Please provide me the document you have for me . 
[/INST]"""

        # 构造 chosen 和 rejected
        chosen_text = "\n\n".join(chosen_final)
        rejected_text = "\n\n".join(rejected_paragraphs)

        # 添加到新数据集
        new_dataset.append({
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text
        })

    # 保存新数据集
    try:
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=4)
        print(f"New dataset has been generated and saved to {dest_path}")
    except Exception as e:
        print(f"Error saving dataset to {dest_path}: {e}")
        exit(1)

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Build MITO dataset from source JSON file.")
    parser.add_argument("--ds-source", type=str, required=True, help="Source JSON dataset path")
    parser.add_argument("--dest-dir", type=str, required=True, help="Destination JSON dataset path")

    # 解析参数
    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.ds_source):
        print(f"Error: Source file {args.ds_source} does not exist!")
        exit(1)

    # 构建并保存数据集
    build_mito_dataset(args.ds_source, args.dest_dir)

if __name__ == "__main__":
    import os
    main()