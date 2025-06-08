import argparse
import json
from datasets import load_dataset
import random
from transformers import AutoTokenizer
import os

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful and reliable assistant. Your task is to provide accurate, well-reasoned answers based on the given documents. Be critical of the information provided and ensure your responses are grounded in the most trustworthy sources."""

# Default user prompt template
DEFAULT_USER_PROMPT_TEMPLATE = """Below is a question and some retrieved documents. Some documents may be irrelevant or contain inaccurate information.
Please analyze the documents carefully and provide a high-quality, concise, and accurate answer based on the reliable information.

[Retrieved Documents]
{paragraph}

[Question]
{question}

Please provide your answer based on the documents above:"""

def format_docs_to_str(doc_list_of_dicts: list[dict]) -> str:
    """
    将文档字典列表（每个字典包含 'title', 'text'）格式化为单一字符串。
    """
    formatted_texts = []
    for i, doc_dict in enumerate(doc_list_of_dicts):
        title = doc_dict.get('title', '').strip()
        text = doc_dict.get('text', '').strip()
        if title:
            formatted_texts.append(f"Title: {title}\n{text}")
        else:
            formatted_texts.append(text)
    return "\n\n".join(filter(None, formatted_texts))

def build_prompt_with_template(question: str, documents_str: str, prompt_template: str) -> str:
    """
    使用模板构建用户消息内容
    """
    return prompt_template.format(
        paragraph=documents_str,
        question=question
    )

def build_chat_prompt(tokenizer: AutoTokenizer, user_content: str, system_prompt: str) -> str:
    """
    使用 tokenizer.apply_chat_template 构建标准化的聊天 prompt
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # 使用 apply_chat_template，add_generation_prompt=True 会添加 assistant 开始标记
    chat_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return chat_prompt

def build_mito_dataset(source_path: str, dest_path: str, tokenizer: AutoTokenizer, system_prompt: str, user_prompt_template: str):
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
            has_answer = ctx.get('hasanswer', False)
            title = ctx.get('title', '')
            text = ctx.get('text', '')
            if not doc_id.startswith('fab') and has_answer:
                 original_docs_dicts.append({"title": title, "text": text})
        
        if not original_docs_dicts:
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
        combined_for_attack_dicts = original_docs_dicts + fabrication_docs_dicts
        random.shuffle(combined_for_attack_dicts) 
        attacked_docs_dicts = combined_for_attack_dicts

        # 格式化文档字符串
        original_docs_str = format_docs_to_str(original_docs_dicts)
        attacked_docs_str = format_docs_to_str(attacked_docs_dicts)

        # 4. 使用模板构建用户消息内容
        user_content_normal = build_prompt_with_template(question, original_docs_str, user_prompt_template)
        user_content_attacked = build_prompt_with_template(question, attacked_docs_str, user_prompt_template)

        # 5. 使用 tokenizer.apply_chat_template 构建标准化的 prompt
        final_prompt_normal = build_chat_prompt(tokenizer, user_content_normal, system_prompt)
        final_prompt_attacked = build_chat_prompt(tokenizer, user_content_attacked, system_prompt)
        
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
    parser = argparse.ArgumentParser(description="Build dataset for MITO training with standardized prompt template.")
    parser.add_argument("--ds-source", type=str, required=True, 
                        help="Source JSON/JSONL dataset path or directory.")
    parser.add_argument("--dest-dir", type=str, required=True, 
                        help="Destination JSONL dataset path.")
    parser.add_argument("--model_name", default='Qwen/Qwen2.5-7B-Instruct', type=str,
                        help="Model name for tokenizer (e.g., Qwen/Qwen2.5-7B-Instruct).")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt for the assistant.")
    parser.add_argument("--user_prompt_template", type=str, default=DEFAULT_USER_PROMPT_TEMPLATE,
                        help="User prompt template with {paragraph} and {question} placeholders.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ds_source):
        print(f"Error: Source path {args.ds_source} does not exist!")
        exit(1)
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        print(f"Successfully loaded tokenizer for {args.model_name}")
    except Exception as e:
        print(f"Error: Could not load tokenizer for {args.model_name}. Error: {e}")
        exit(1)

    build_mito_dataset(args.ds_source, args.dest_dir, tokenizer, args.system_prompt, args.user_prompt_template)

if __name__ == "__main__":
    main()