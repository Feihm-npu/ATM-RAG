import os
import argparse
from datasets import load_dataset
import torch
# transformers.AutoModel is used, assuming it's Unsloth's FastLanguageModel or compatible
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from peft import PeftModel
from transformers.trainer_utils import is_main_process
# from accelerate import Accelerator # Added Accelerator

# Ensure this import points to the file containing your minimal_MITOTrainer class
# from min_mito_0523 import minimal_MITOTrainer, MITOConfig
# For this example, let's assume minimal_MITOTrainer and MITOConfig are defined elsewhere
# and are compatible with Hugging Face Trainer/DPOTrainer.
# If they are in the same file, ensure their definitions are present.
# We'll use placeholder classes if the actual import is not available.

# Placeholder for MITOConfig if not imported (replace with actual import)
from min_mito_0523 import MITOConfig, minimal_MITOTrainer

from transformers.utils import logging
logger = logging.get_logger(__name__)


from typing import Dict # Ensure Dict is imported

def main():
    # Initialize Accelerator
    # gradient_accumulation_steps will be handled by the Trainer via TrainingArguments
    


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MITO model')
    parser.add_argument('--model_name', type=str, required=True, help='Base model name or path')
    parser.add_argument('--adapter', type=str, required=True, help='Adapter')
    parser.add_argument('--dataset_name', type=str, required=True, help='Path to the training dataset JSON file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device for training')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of training steps (overrides num_train_epochs if > 0)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saved model and logs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", help='Learning rate scheduler type')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--beta_mito', type=float, default=0.1, help='Beta (alpha) for the KL term in MITO loss')
    parser.add_argument('--max_seq_length_config', type=int, default=2048, help='Max sequence length for model loading (Unsloth FastLanguageModel)')
    parser.add_argument('--dpo_max_length', type=int, default=1024, help='Max total length for D+a sequences in DPOConfig')
    parser.add_argument('--dpo_max_prompt_length', type=int, default=512, help='Max length for answer (a) in DPOConfig (used by tokenize_row)')
    parser.add_argument('--dpo_max_completion_length', type=int, default=512, help='Max length for contexts (D, D_prime) in DPOConfig (used by tokenize_row)')
    parser.add_argument('--logging_steps', type=int, default=1, help='Log every N steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    # Add any other arguments your script might need

    args = parser.parse_args()

    # Load model and tokenizer using Unsloth's FastLanguageModel (assumed by AutoModel usage)
    # The Trainer will handle moving the model to the correct device per process via accelerator
    logger.info(f"Loading model: {args.model_name}")


    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        attn_implementation="flash_attention_2",
        # torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        is_trainable=True,
    )
    ref_model = model

    logger.info("Model loaded.")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Set tokenizer.pad_token to tokenizer.eos_token")

    # 1. Load your dataset
    # Dataset loading should be fine as is, each process will load it.
    # Alternatively, load on main process and broadcast, but load_dataset is often efficient.
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset_dict = load_dataset('json', data_files=args.dataset_name)
    
    # For debugging or quick runs, you might want to select a smaller portion.
    # Ensure this range is valid for your dataset size.
    train_dataset = dataset_dict["train"]
    logger.info(f"Loaded {len(train_dataset)} samples for training.")


    # 2. Rename columns to match what minimal_MITOTrainer.tokenize_row will expect
    column_mapping = {
        'answer': 'prompt',
        'prompt': 'chosen',
        'adv_prompt': 'rejected'
    }
    # Check if columns exist before renaming to avoid errors
    current_columns = train_dataset.column_names
    actual_mapping = {k: v for k, v in column_mapping.items() if k in current_columns}
    if len(actual_mapping) < len(column_mapping):
        logger.info(f"Warning: Not all expected columns for renaming found. Found: {current_columns}. Expected to rename: {list(column_mapping.keys())}")
    
    if actual_mapping:
        train_dataset = train_dataset.rename_columns(actual_mapping)
        logger.info(f"Renamed dataset columns. New features: {train_dataset.features}")
    else:
        logger.info(f"No columns were renamed. Current features: {train_dataset.features}")


    # Configure DPO training arguments (used by minimal_MITOTrainer)
    # MITOConfig should be a subclass of TrainingArguments or DPOConfig
    dpo_config = MITOConfig(
        per_device_train_batch_size=args.batch_size,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        bf16=True,
        # num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else 1e6, # Effectively infinite if max_steps is used
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        # fp16=accelerator.use_fp16, # Trainer will handle mixed precision based on accelerator config
        # bf16=accelerator.use_bf16, # Or explicit TrainingArguments like fp16=True/bf16=True
        logging_steps=args.logging_steps,
        optim="adamw_8bit", # Unsloth recommends adamw_8bit; ensure it's compatible with your model/setup
        seed=42,
        output_dir=args.output_dir,
        
        # MITO/DPO specific parameters
        beta=args.beta_mito,
        
        max_length=args.dpo_max_length,
        max_prompt_length=args.dpo_max_prompt_length,
        gradient_checkpointing=True,
        # max_completion_length=args.dpo_max_completion_length,

        remove_unused_columns=False,
        sft_on_d_prime=False, # Assuming this is a custom arg for MITOConfig
        # report_to="wandb", # If you use wandb, configure it in TrainingArguments
        #gradient_checkpointing=True, # Consider for large models to save memory
    )

    # Initialize and train the minimal_MITOTrainer
    # The Trainer will use the accelerator object implicitly for distributed training.
    mito_trainer = minimal_MITOTrainer(
        model=model, # The model is passed as is; Trainer + Accelerate will prepare it.
        # ref_model=ref_model, # If you have a reference model, pass it here.
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=train_dataset, # Replace with a proper validation set if available
        processing_class=tokenizer, # Pass the tokenizer here (standard for DPOTrainer)
                             # Original code had 'processing_class=tokenizer'. Adjusted to 'tokenizer'.
                             # If your minimal_MITOTrainer specifically needs 'processing_class', revert this.
        # data_collator=None, # Defaults to DataCollatorForPreference, which should be fine
        
        # mito_alpha=args.beta_mito
    )
    logger.info("minimal_MITOTrainer initialized.")

    logger.info("Starting MITO training...")
    mito_trainer.train()
    logger.info("MITO training finished.")
    
    # Save the final model
    # Ensure all processes are synchronized before saving.

    final_save_path = os.path.join(args.output_dir, "model_mito_final")
    
    # Save model and tokenizer only on the main process
    logger.info(f"Saving model to {final_save_path}...")
    # DPOTrainer.save_model() should handle PEFT adapters correctly.
    # If not using PEFT, it saves the full model.
    # The model passed to save_model should be the original one if it was wrapped.
    # However, trainer.save_model() is designed to handle this.
    # If you were saving manually:
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(final_save_path)
    if is_main_process():
        mito_trainer.save_model(final_save_path) 
        tokenizer.save_pretrained(final_save_path)
        logger.info(f"Successfully completed training. Model saved to: {final_save_path}")
    

if __name__ == "__main__":
    main()
