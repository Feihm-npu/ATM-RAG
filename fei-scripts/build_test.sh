#!/bin/bash
export NVIDIA_VISIBLE_DEVICES=4,5  # 确保变量对 Python 进程可见
export CUDA_VISIBLE_DEVICES=4,5  # vLLM 通常需要 CUDA 变量

mkdir -p /home/feihm/llm-fei/Data/ATM/test_data_with_fabs
mkdir -p /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/ask_output

echo "First part started!"
python build_ask_gpt.py \
    --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --world_size 2 \
    --ds_name NQ/NQ \
    --dest_dir /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/ask_output

echo "First part completed!"

python fab_merge.py \
    --ds_name NQ/NQ \
    --dest_dir /home/feihm/llm-fei/Data/ATM/test_data_with_fabs

echo "Second part completed!"


# python generator_sft_data_prepare.py