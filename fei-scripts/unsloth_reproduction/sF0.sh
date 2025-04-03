export CUDA_VISIBLE_DEVICES=6,7  # vLLM 通常需要 CUDA 变量

RUNs=1

mkdir -p /home/feihm/llm-fei/Data/ATM-RAG/test_data_with_fabs1/NQ

echo "First part started!"
OMP_NUM_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/sF0_generator_fake.py \
    --model_name /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s2_experiments/merged_for_vllm_final \
    --world_size 2 \
    --ds_name NQ/NQ \
    --dest_dir "/home/feihm/llm-fei/Data/ATM/test_data_with_fabs/"


echo "First part completed!"

# echo "Second part started!"

# python fab_merge.py \
#     --ds_name NQ/NQ \
#     --dest_dir /home/feihm/llm-fei/Data/ATM/test_data_with_fabs

# echo "Second part completed!"