export CUDA_VISIBLE_DEVICES=0

OMP_NUM_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=1 s1_tuning_generator.py \
    --model_name_or_path unsloth/Qwen2.5-7B \
    --train_data /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/NQ/attacked_train_fab_for_sft_arrows \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --bf16 \
    --output_dir ./s1_experiments/ \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --wandb_project atm-sft-unsloth
