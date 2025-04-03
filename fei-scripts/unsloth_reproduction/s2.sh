export CUDA_VISIBLE_DEVICES=1

OMP_NUM_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python s3_adversary_dpo.py \
  --model_name unsloth/Qwen2.5-7B \
  --train_file /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/NQ/NQ_dpo.jsonl \
  --output_dir ./s2_experiments/model_1epoch_dpo \
  --wandb_project unsloth-s2 \
  --bf16
