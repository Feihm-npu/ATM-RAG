#!/bin/bash
## This script is used for MITO training the generator.

export CUDA_VISIBLE_DEVICES=4
# 运行脚本示例
OMP_NUM_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python dpo_test.py \
    --model_name "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_experiments/for_vllm" \
    --dataset_name "ZSvedic/gpt4o-arena-brevity-dpo" \
    --batch_size 10 \
    --num_train_epochs 1 \
    --output_dir "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_experiments/output"