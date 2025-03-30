#!/bin/bash
export NVIDIA_VISIBLE_DEVICES=0  # 确保变量对 Python 进程可见
export CUDA_VISIBLE_DEVICES=0  # vLLM 通常需要 CUDA 变量

epoch=1

mkdir -p /home/feihm/llm-fei/Data/ATM/test_data_with_fabs1/NQ
mkdir -p /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/ask_output1/NQ

echo "First part started!"
OMP_NUM_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_build_data/build_ask_gpt.py \
    --model_name /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s2_experiments/merged_for_vllm_final \
    --world_size 1 \
    --ds_name NQ/NQ \
    --dest_dir "/home/feihm/llm-fei/Data/ATM/test_data_with_fabs/ask_output1"

# OMP_NUM_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s0_data.py \
#     --model_dir /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s2_experiments/model_1epoch_dpo \
#     --ds_name NQ/NQ \
#     --bz 28 \
#     --dest_dir "/home/feihm/llm-fei/Data/ATM/test_data_with_fabs/ask_output{$epoch}"\
#     --num_dups 5 \
#     --bf16

echo "First part completed!"

# python fab_merge.py \
#     --ds_name NQ/NQ \
#     --dest_dir /home/feihm/llm-fei/Data/ATM/test_data_with_fabs{$epoch}

# echo "Second part completed!"


# python generator_sft_data_prepare.py