import argparse
import json
from datasets import load_dataset
import random
from transformers import AutoTokenizer
import os

# Qwen2 specific tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

def format_docs_to_str(doc_list_of_dicts: list[dict]) -> str:
    """
    将文档字典列表（每个字典包含 'title', 'text'）格式化为单一字符串。
    """
    formatted_texts = []
    # 根据论文，ATM的Generator输入的是拼接的documents
    # d1⊕...⊕dn (Section 3.2)
    # 这里的格式可以根据实际模型输入调整，简单拼接文本内容
    for i, doc_dict in enumerate(doc_list_of_dicts):
        # formatted_texts.append(f"Document {i+1}:\nTitle: {doc_dict['title']}\nText: {doc_dict['text']}")
        # 论文中Figure 2 和 Figure 3 的例子是 [TITLE] content [TEXT] content
        # 但通常直接拼接文本内容更常见
        title = doc_dict.get('title', '').strip()
        text = doc_dict.get('text', '').strip()
        if title:
            formatted_texts.append(f"Title: {title}\n{text}")
        else:
            formatted_texts.append(text)
    return "\n\n".join(filter(None, formatted_texts)) # 过滤掉空字符串

def build_mito_dataset(source_path: str, dest_path: str, tokenizer: AutoTokenizer, system_message: str = DEFAULT_SYSTEM_MESSAGE):
    """
    根据MITO论文和官方mito_tokenize_row的思路构建数据集。
    输出每个样本包含: 'answer', 'prompt', 'adv_prompt'
    """
    try:
        if os.path.isdir(source_path):
            data_files = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith((".json", ".jsonl"))]
            if not data_files:
                print(f"Error: No JSON or JSONL files found in directory {source_path}")
                exit(1)
            dataset = load_dataset("json", data_files=data_files, split="train")
        elif os.path.isfile(source_path):
            dataset = load_dataset("json", data_files=source_path, split="train")
        else:
            print(f"Error: Source path {source_path} is not a valid file or directory.")
            exit(1)
    except Exception as e:
        print(f"Error loading dataset from {source_path}: {e}")
        exit(1)

    new_dataset_for_jsonl = []
    skipped_samples_no_answer = 0
    skipped_samples_no_original_docs = 0

    for sample_idx, sample in enumerate(dataset):
        question = sample.get('question')
        if not question or not isinstance(question, str):
            print(f"Warning: Skipping sample {sample_idx} due to missing or invalid question.")
            continue
            
        answers = sample.get('answers')
        if not answers or not isinstance(answers, list) or len(answers) == 0:
            skipped_samples_no_answer += 1
            continue
        golden_answer_str = answers[0]
        if not isinstance(golden_answer_str, str) or not golden_answer_str.strip():
            skipped_samples_no_answer += 1
            continue

        ctxs = sample.get('ctxs', [])

        # 1. 构建原始文档列表 D (original_docs_dicts)
        original_docs_dicts = []
        for ctx in ctxs:
            if not isinstance(ctx, dict): continue
            doc_id = ctx.get('id', '')
            has_answer = ctx.get('hasanswer', False) # 论文中D是包含golden answer的文档
            title = ctx.get('title', '')
            text = ctx.get('text', '')
            # 根据论文，D是包含有用信息的文档集合 (useful documents)
            # ATM Figure 1暗示D是golden knowledge.
            # RAFT论文 (被ATM引用) 中，D是oracle (golden) documents.
            # 我们假设 id 不以 "fab" 开头且 hasanswer 为 True 的是原始好文档
            if not doc_id.startswith('fab') and has_answer:
                 original_docs_dicts.append({"title": title, "text": text})
        
        if not original_docs_dicts: # 如果没有包含答案的原始文档，无法构成D，跳过
            skipped_samples_no_original_docs += 1
            continue

        # 2. 构建伪造文档列表 d' (fabrication_docs_dicts)
        fabrication_docs_dicts = []
        for ctx in ctxs:
            if not isinstance(ctx, dict): continue
            doc_id = ctx.get('id', '')
            title = ctx.get('title', '')
            text = ctx.get('text', '')
            if doc_id.startswith('fab'):
                fabrication_docs_dicts.append({"title": title, "text": text})

        # 3. 构建被攻击文档列表 D' (attacked_docs_dicts)
        #    D' = LP[D U d'] (List Permutation of D union d')
        #    论文 Section 3.1: D' = LP[D U {d'}] where {d'} is the set of generated fabrications
        combined_for_attack_dicts = original_docs_dicts + fabrication_docs_dicts
        random.shuffle(combined_for_attack_dicts) 
        attacked_docs_dicts = combined_for_attack_dicts
        
        # 如果没有任何文档（原始+伪造），则无法构建 D'，这种情况理论上不会发生，因为 original_docs_dicts 非空
        if not attacked_docs_dicts: 
            # Fallback or skip, though D' should at least contain D
            # For MITO, D' is crucial for SFT loss. If it's empty, it's problematic.
            # Since original_docs_dicts is guaranteed non-empty, attacked_docs_dicts will also be non-empty.
            pass


        original_docs_str = format_docs_to_str(original_docs_dicts)
        attacked_docs_str = format_docs_to_str(attacked_docs_dicts)

        # 4. 构建 'prompt' 和 'adv_prompt' 字符串
        #    这些字符串将作为 mito_tokenize_row 函数的输入
        #    它们是答案生成前的完整上下文，遵循Qwen2聊天模板

        # 'prompt' for normal context (D)
        # 用户回合包含问题和相关文档
        user_content_normal = f"{question}\n\nHere are some documents that might be relevant:\n{original_docs_str}"
        prompt_normal_parts = [
            f"{IM_START}system\n{system_message}{IM_END}\n",
            f"{IM_START}user\n{user_content_normal}{IM_END}\n",
            f"{IM_START}assistant\n" 
        ]
        final_prompt_normal = "".join(prompt_normal_parts)

        # 'adv_prompt' for attacked context (D')
        user_content_attacked = f"{question}\n\nHere are some documents that might be relevant:\n{attacked_docs_str}"
        prompt_attacked_parts = [
            f"{IM_START}system\n{system_message}{IM_END}\n",
            f"{IM_START}user\n{user_content_attacked}{IM_END}\n",
            f"{IM_START}assistant\n"
        ]
        final_prompt_attacked = "".join(prompt_attacked_parts)
        
        new_dataset_for_jsonl.append({
            "answer": golden_answer_str,
            "prompt": final_prompt_normal,
            "adv_prompt": final_prompt_attacked,
        })

    print(f"Total samples processed from source: {len(dataset)}")
    print(f"Skipped samples due to no/invalid answer: {skipped_samples_no_answer}")
    print(f"Skipped samples due to no original (hasanswer=True) documents for D: {skipped_samples_no_original_docs}")
    print(f"Number of samples in the new dataset: {len(new_dataset_for_jsonl)}")

    try:
        with open(dest_path, "w", encoding="utf-8") as f:
            for entry in new_dataset_for_jsonl:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"New dataset has been generated and saved to {dest_path}")
    except Exception as e:
        print(f"Error saving dataset to {dest_path}: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Build dataset for MITO training, aligning with official tokenization.")
    parser.add_argument("--ds-source", type=str, required=True, 
                        help="Source JSON/JSONL dataset path or directory.")
    parser.add_argument("--dest-dir", type=str, required=True, 
                        help="Destination JSONL dataset path.")
    parser.add_argument("--model_name", default='Qwen/Qwen2-7B-Instruct', type=str,
                        help="Model name for tokenizer (e.g., for special tokens).")
    parser.add_argument("--system_message", type=str, default=DEFAULT_SYSTEM_MESSAGE,
                        help="System message to use in the prompt.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ds_source):
        print(f"Error: Source path {args.ds_source} does not exist!")
        exit(1)
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load tokenizer for {args.model_name}. Using default Qwen2 tokens. Error: {e}")
        tokenizer = None 

    build_mito_dataset(args.ds_source, args.dest_dir, tokenizer, args.system_message)

if __name__ == "__main__":
    main()
