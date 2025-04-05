import os
import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
import torch
from transformers import TrainingArguments
from trl import DPOConfig
from min_mito import min_MITOTrainer

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train DPO model with specified parameters')
    parser.add_argument('--model_name', type=str, required=True, help='Base model name or path')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per device')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=10, help='Maximum number of training steps (overrides num_train_epochs if set)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saved model')
    
    args = parser.parse_args()

    # 设置GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    max_seq_length = 2048

    # 加载模型和tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # Actually, the model has already been pre-trained.
    # 添加LoRA权重
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r = 64,
    #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                      "gate_proj", "up_proj", "down_proj",],
    #     lora_alpha = 64,
    #     lora_dropout = 0,
    #     bias = "none",
    #     use_gradient_checkpointing = "unsloth",
    #     random_state = 3407,
    #     max_seq_length = max_seq_length,
    # )

    # 加载数据集
    # dataset = load_dataset('json', data_files=args.dataset_name, split='train')
    dataset = load_dataset('json', data_files=args.dataset_name)
    # dataset["test"] = dataset["test"].select(range(20))

    # 配置DPO训练参数
    dpo_config = DPOConfig(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = args.num_train_epochs,
        # max_steps = args.max_steps if args.max_steps > 0 else None,  # 如果max_steps大于0，则使用它
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir = args.output_dir,
        beta = 0.1,
        max_length = 1024,
        max_prompt_length = 1024,
    )

    # 初始化并训练DPO trainer
    dpo_trainer = min_MITOTrainer(
        model = model,
        ref_model = None,
        args = dpo_config,
        train_dataset = dataset["train"],
        eval_dataset = dataset["train"],
        processing_class = tokenizer,
    )

    dpo_trainer.train()
    
    # 保存模型
    save_path = os.path.join(args.output_dir, "model_mito_final")
    dpo_trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"[✓] 训练完成，模型已保存至：{save_path}")

if __name__ == "__main__":
    main()