#!/bin/bash

# 运行脚本示例
python merge_lora2vllm.py \
    --base_model_name "unsloth/Qwen2.5-7B-Instruct" \
    --adapter_dir "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/ATM_RAG_0524/gen_model/model_mito_final" \
    --output_dir "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/ATM_RAG_0524/forEval"