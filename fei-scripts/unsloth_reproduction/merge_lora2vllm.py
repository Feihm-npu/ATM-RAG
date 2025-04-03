import torch
import argparse
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from peft import PeftModel

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Merge LoRA weights into base model')
    parser.add_argument('--base_model_name', type=str, required=True, help='Base model name or path')
    parser.add_argument('--adapter_dir', type=str, required=True, help='Directory containing adapter weights')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for merged model')
    
    args = parser.parse_args()

    # Step 1: 加载 base 模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model_name,
        max_seq_length = 4096,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )

    # Step 3: 使用 PeftModel 的方法合并 LoRA 权重
    model = PeftModel.from_pretrained(model, args.adapter_dir)

    # Step 4: 保存为标准 HuggingFace 格式
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[✓] 已成功将 LoRA 模型合并并保存为 vLLM 可用格式：{args.output_dir}")

if __name__ == "__main__":
    main()