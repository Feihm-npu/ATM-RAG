import os
import unsloth
from unsloth import FastLanguageModel
# PatchDPOTrainer()
from datasets import load_dataset
import torch
from transformers import TrainingArguments, AutoModelForCausalLM

from min_mito import min_MITOTrainer, mito_tokenize_row
from mito import MITODataCollatorWithPadding  # 你已定义的 data_collator
from trl.trainer import DPOConfig
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -------------------- Load Base Model (Qwen2.5-7B) --------------------
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B",
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    max_seq_length = max_seq_length,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# -------------------- Load Dataset --------------------
dataset = load_dataset("ZSvedic/gpt4o-arena-brevity-dpo")
dataset["train"] = dataset["test"].select(range(20))

# rename keys for compatibility
dataset = dataset.rename_columns({
    "prompt": "prompt",
    "chosen": "answer",
    "rejected": "adv_prompt",
})

# tokenize with your mito_tokenize_row
tokenized_dataset = dataset["train"].map(lambda x: mito_tokenize_row(x, tokenizer))

# Dummy placeholders to bypass DPO required keys (not used in MITO loss)
tokenized_dataset = tokenized_dataset.add_column("chosen", [""] * len(tokenized_dataset))
tokenized_dataset = tokenized_dataset.add_column("rejected", [""] * len(tokenized_dataset))

# -------------------- Reference Model --------------------
ref_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-7B",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    device_map="auto",
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# -------------------- TrainingArguments --------------------
training_args = DPOConfig(
    output_dir="./outputs_min_mito",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",  # wandb if needed
    save_strategy="no",
    remove_unused_columns=False,
)

# -------------------- Trainer --------------------
trainer = min_MITOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    processing_class=tokenizer,
    # data_collator=MITODataCollatorWithPadding(
    #     pad_token_id=tokenizer.pad_token_id,
    #     label_pad_token_id=-100,
    #     is_encoder_decoder=False
    # ),
    loss_type="sigmoid",  # for DPO loss
    beta=0.1,  # KL loss weight
)

# -------------------- Training --------------------
trainer.train()

trainer.save_model("./outputs_min_mito/final_model")
tokenizer.save_pretrained("./outputs_min_mito/final_model")
