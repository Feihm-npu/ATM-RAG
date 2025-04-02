import os
from datasets import load_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Optional set GPU device ID

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
# PatchDPOTrainer()
import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from min_mito import min_MITOTrainer
from trl import DPOConfig

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/zephyr-sft-bnb-4bit",
    model_name = "unsloth/Qwen2.5-7B",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
)

dataset = load_dataset("ZSvedic/gpt4o-arena-brevity-dpo")
dataset["test"] = dataset["test"].select(range(20))

dpo_trainer = min_MITOTrainer(
    model = model,
    ref_model = None,
    args = DPOConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir = "outputs",
        beta = 0.1,
        max_length = 1024,
        max_prompt_length = 512,
    ),
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    processing_class = tokenizer,
)
dpo_trainer.train()