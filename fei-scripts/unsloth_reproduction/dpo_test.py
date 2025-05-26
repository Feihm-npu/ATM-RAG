import os
import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
import torch
from transformers import TrainingArguments, AutoTokenizer # Added AutoTokenizer for clarity
# from trl import DPOConfig

# Ensure this import points to the file containing your minimal_MITOTrainer class
from min_mito_0523 import minimal_MITOTrainer, MITOConfig
# If minimal_MITOTrainer is in the same file, you can comment out the above
# and ensure the class definition is available.

from typing import Dict # Ensure Dict is imported if not already via other imports

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MITO model with specified parameters')
    parser.add_argument('--model_name', type=str, required=True, help='Base model name or path')
    parser.add_argument('--dataset_name', type=str, required=True, help='Path to the training dataset JSON file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per device for training')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=-1, help='Maximum number of training steps (overrides num_train_epochs if > 0)')
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

    args = parser.parse_args()

    # Unsloth: Apply DPO Patch
    # This might patch the base DPOTrainer, which minimal_MITOTrainer inherits from.
    PatchDPOTrainer()

    # Load model and tokenizer using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length_config,
        dtype=None,  # Unsloth handles dtype
        load_in_4bit=True, # Or False, depending on your setup
    )
    
    # Important: Set pad_token if not already set.
    # For decoder-only models, it's common to use eos_token as pad_token.

    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer.pad_token to tokenizer.eos_token")
    
    # It's generally good practice for decoder-only models to have padding on the left
    # if the model architecture benefits from it or if specific utilities expect it.
    # However, `flush_left` in DPOTrainer's `concatenated_forward` handles this for the combined sequence.
    # Our `pad_to_length` in `concatenated_inputs` also specifies padding side.

    # 1. Load your dataset
    dataset_dict = load_dataset('json', data_files=args.dataset_name)
    train_dataset = dataset_dict["train"] # Assuming your JSON has a "train" key or it's a single file

    # 2. Rename columns to match what minimal_MITOTrainer.tokenize_row will expect
    # Expected by tokenize_row: "prompt" (for 'a'), "chosen" (for 'D'), "rejected" (for 'D_prime')
    column_mapping = {
        'answer': 'prompt',     # Your 'answer' (a) becomes "prompt" for tokenize_row
        'prompt': 'chosen',     # Your 'prompt' (context D) becomes "chosen"
        'adv_prompt': 'rejected' # Your 'adv_prompt' (context D') becomes "rejected"
    }
    train_dataset = train_dataset.rename_columns(column_mapping)
    print(f"Renamed dataset columns. New features: {train_dataset.features}")


    # Configure DPO training arguments (used by minimal_MITOTrainer)
    # Note: DPOConfig is a subclass of TrainingArguments
    dpo_config = MITOConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else 1e6, # Effectively infinite if max_steps is used
        max_steps=args.max_steps if args.max_steps > 0 else -1, # Use -1 if not set, so num_train_epochs is primary
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit", # Unsloth recommends adamw_8bit
        seed=42,
        output_dir=args.output_dir,
        
        # MITO/DPO specific parameters
        beta=args.beta_mito,  # This is the 'alpha' for the KL term in MITO loss
        
        # Length controls for tokenization and model processing:
        # max_length: Total length for the combined sequence (e.g., D+a)
        # max_prompt_length: Used by our tokenize_row for the 'answer' (a)
        # max_completion_length: Used by our tokenize_row for 'contexts' (D, D_prime)
        max_length=args.dpo_max_length,
        max_prompt_length=args.dpo_max_prompt_length,
        max_completion_length=args.dpo_max_completion_length, # Explicitly set for clarity
        # max_target_length=args.dpo_max_completion_length, # Alias for max_completion_length

        remove_unused_columns=False, # Important for custom dataset structures
        sft_on_d_prime=False,
        # report_to="wandb", # If you use wandb
    )

    # Initialize and train the minimal_MITOTrainer
    mito_trainer = minimal_MITOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=train_dataset, # Replace with a proper validation set if available
        processing_class=tokenizer, # Pass the tokenizer here
        # data_collator=None, # Defaults to DataCollatorForPreference, which should be fine
    )

    print("Starting MITO training...")
    mito_trainer.train()
    
    # Save the final model
    final_save_path = os.path.join(args.output_dir, "model_mito_final")
    # If using LoRA, Unsloth might have a specific way to save adapters,
    # or model.save_pretrained for the full model if merged.
    # DPOTrainer.save_model() should handle PEFT adapters correctly.
    mito_trainer.save_model(final_save_path) 
    tokenizer.save_pretrained(final_save_path)

    print(f"Successfully completed training. Model saved to: {final_save_path}")

if __name__ == "__main__":
    main()
# ```

# **Key changes and considerations in this script:**

# 1.  **Import:** `from min_mito_0523 import minimal_MITOTrainer` is used. Ensure this file path is correct.
# 2.  **Tokenizer `pad_token`:** Added a check to set `tokenizer.pad_token = tokenizer.eos_token` if it's not already set. This is crucial for decoder-only models.
# 3.  **`DPOConfig` Length Parameters:**
#     * I've added command-line arguments for `dpo_max_length`, `dpo_max_prompt_length`, and `dpo_max_completion_length` to give you explicit control.
#     * `dpo_config.max_prompt_length` will control the max length of your 'answer' ($a$) in `tokenize_row`.
#     * `dpo_config.max_completion_length` will control the max length of your 'contexts' ($D, D'$) in `tokenize_row`.
#     * `dpo_config.max_length` controls the total length of the combined sequence ($D+a$) that the model processes.
# 4.  **`max_steps` Handling:** Adjusted `num_train_epochs` and `max_steps` in `DPOConfig` for more standard behavior. If `max_steps` is positive, it takes precedence.
# 5.  **`remove_unused_columns=False`:** Added to `DPOConfig`. This is often important when working with custom dataset processing, as the default `True` might remove columns your `tokenize_row` or subsequent steps expect before they are transformed into the final model inputs.
# 6.  **Unsloth `PatchDPOTrainer()`:** Kept this as it's part of the Unsloth workflow and might be necessary for compatibility with `FastLanguageModel`.
# 7.  **Saving Model:** `dpo_trainer.save_model()` is used, which should correctly save PEFT adapters if you're using LoRA with `FastLanguageModel`.

# To run this script, you would execute it from your terminal, providing the necessary arguments:

# ```bash
# python dpo_test.py \
#     --model_name "your_base_model_name_or_path" \
#     --dataset_name "path_to_your_data.json" \
#     --output_dir "./mito_output" \
#     --batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --num_train_epochs 1 \
#     --learning_rate 2e-5 \
#     --beta_mito 0.1 \
#     --dpo_max_length 1024 \
#     --dpo_max_prompt_length 512 \
#     --dpo_max_completion_length 512 \
#     # ... other arguments if needed
# ```

# Make sure the `min_mito_0523.py` file containing the `minimal_MITOTrainer` class is in the same directory as `dpo_test.py` or accessible in your `PYTHONPAT