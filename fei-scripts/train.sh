export NVIDIA_VISIBLE_DEVICES=4,0
export CUDA_VISIBLE_DEVICES=4,0
# export LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LIBRARY_PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

MASTER_ADDR=$(hostname -I | awk '{print $1}')
# --hostfile='hosts.cfg'
OMP_NUM_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4,0 deepspeed  /home/feihm/llm-fei/ATM-RAG/atm_train/generator_sft/train.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --train_data /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/NQ/attacked_train_fab_for_sft_arrows \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4  \
    --bf16 \
    --deepspeed_file /home/feihm/llm-fei/ATM-RAG/atm_train/generator_sft/ds_cfg.json \
    --output_dir ./experiments/ \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --learning_rate 5e-6