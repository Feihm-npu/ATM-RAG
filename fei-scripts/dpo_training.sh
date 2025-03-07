export CUDA_VISIBLE_DEVICES=0,4,5 
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OMP_NUM_THREADS=64 accelerate launch --config_file /home/feihm/llm-fei/ATM-RAG/atm_train/mito/acc.yaml --main_process_port 2951 \
    /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_dpo/train_dpo.py \
    --model_name_or_path /home/feihm/llm-fei/ATM-RAG/fei-scripts/experiments/model_final/ \
    --train_data /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/NQ/NQ_dpo.jsonl \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --learning_rate 8e-7 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --output_dir ./experiments \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_length 4096 \
    --max_prompt_length 3072
