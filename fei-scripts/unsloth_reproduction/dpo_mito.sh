#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 
# 运行脚本示例
OMP_NUM_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python dpo_test.py \
    --model_name "unsloth/Qwen2.5-7B" \
    --dataset_name "ZSvedic/gpt4o-arena-brevity-dpo" \
    --batch_size 4 \
    --num_train_epochs 3 \
    --output_dir "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_experiments/output"