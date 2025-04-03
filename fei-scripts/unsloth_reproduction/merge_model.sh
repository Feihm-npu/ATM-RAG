#!/bin/bash

# 运行脚本示例
python merge_lora2vllm.py \
    --base_model_name "unsloth/Qwen2.5-7B" \
    --adapter_dir "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_experiments/model_final" \
    --output_dir "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_experiments/for_vllm"