## This script is used for training the attacker model with DPO.

export CUDA_VISIBLE_DEVICES=1

OMP_NUM_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python s3_adversary_dpo.py \
  --model_name unsloth/Qwen2.5-7B \
  --train_file /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/NQ/NQ_dpo.jsonl \
  --output_dir ./s2_experiments/model_dpo \
  --wandb_project unsloth-s3-attacker-dpo \
  --bf16


## After the model fine-tuning, the model and tokenizer will be saved in the output directory.
## However, if you want to use the model in the future, you need to load the model and tokenizer from the output directory.
## In order to use vllm for inference, you need to convert the model to vllm format as follows:


# python merge_lora2vllm.py \
#     --base_model_name "unsloth/Qwen2.5-7B" \
#     --adapter_dir "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s2_experiments/model_dpo" \
#     --output_dir "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s2_experiments/model_dpo"



## ATM-RAG/fei-scripts/unsloth_reproduction/merge_model.sh