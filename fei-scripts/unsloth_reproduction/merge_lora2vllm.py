import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
# Step 1: 加载 base 模型
base_model_name = "unsloth/Qwen2.5-7B"
adapter_dir = "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_experiments/model_final"
output_dir = "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_experiments/for_vllm"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_name,
    max_seq_length = 4096,
    dtype = torch.bfloat16,
    load_in_4bit = False,
)



# ✅ Step 3: 使用 PeftModel 的方法合并 LoRA 权重
from peft import PeftModel
model = PeftModel.from_pretrained(model, adapter_dir)

# ✅ Step 4: 保存为标准 HuggingFace 格式
merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"[✓] 已成功将 LoRA 模型合并并保存为 vLLM 可用格式：{output_dir}")
