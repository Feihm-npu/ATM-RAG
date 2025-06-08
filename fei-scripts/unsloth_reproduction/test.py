from transformers import AutoModelForCausalLM, AutoConfig
import os

model_dir = "./ATM_RAG_0603/adv_model"  # 你的保存路径

model = AutoModelForCausalLM.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)

vocab_from_weights = model.get_input_embeddings().weight.shape[0]
vocab_from_config = config.vocab_size

print(f"🔍 权重中的 vocab size（embedding 行数）: {vocab_from_weights}")
print(f"📄 config.json 中的 vocab size:         {vocab_from_config}")

if vocab_from_weights != vocab_from_config:
    print("❌ vocab_size 不一致，会导致 vLLM 报错！")
else:
    print("✅ vocab_size 一致，理论上 vLLM 可正常加载。")
