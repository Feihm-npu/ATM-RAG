## This script is used for initial fine-tuning the generator, only need to be run once.

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

## After the model fine-tuning, the model and tokenizer will be saved in the output directory.
## However, if you want to use the model in the future, you need to load the model and tokenizer from the output directory.
## In order to use vllm for inference, you need to convert the model to vllm format as follows:


# python merge_lora2vllm.py \
#     --base_model_name "unsloth/Qwen2.5-7B" \
#     --adapter_dir "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_experiments/model_final" \
#     --output_dir "/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_experiments/for_vllm"



## ATM-RAG/fei-scripts/unsloth_reproduction/merge_model.sh