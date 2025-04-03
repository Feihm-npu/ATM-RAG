export NVIDIA_VISIBLE_DEVICES=4  # 确保变量对 Python 进程可见
export CUDA_VISIBLE_DEVICES=4  # vLLM 通常需要 CUDA 变量


OMP_NUM_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s4_mito.py \
    --model_name_or_path /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s2_experiments/model_1epoch_dpo \
    --train_data /home/feihm/llm-fei/Data/ATM/test_data_with_fabs_updated/NQ/NQ_fab.jsonl \
    --output_dir /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s4_experiments/ \
    --bf16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --wandb_project mito-dpo-run
