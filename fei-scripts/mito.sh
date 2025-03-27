export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


accelerate launch --main_process_port 2950 --config_file /home/feihm/llm-fei/ATM-RAG/atm_train/mito/acc.yaml /home/feihm/llm-fei/ATM-RAG/atm_train/mito/train_with_mito.py \
    --model_name_or_path /home/feihm/llm-fei/ATM-RAG/fei-scripts/experiments/model_final/ \
    --train_data /home/feihm/llm-fei/Data/ATM/test_data_with_fabs_updated/NQ/NQ_fab.jsonl \
    --beta 0.2 \
    --gradient_accumulation_steps 8 \
    --max_steps 10 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --output_dir ./experiments \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_length 2048 \
    --max_prompt_length 2048