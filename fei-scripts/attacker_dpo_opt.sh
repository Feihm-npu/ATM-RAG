# epoch_suffix=${1}
export CUDA_VISIBLE_DEVICES=4,0

jsonl_root=/home/feihm/llm-fei/Data/ATM/test_data_with_fabs/NQ


OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=4,0 torchrun --nnodes=1 --nproc_per_node=2 \
    /home/feihm/llm-fei/ATM-RAG/atm_train/ppl_infer/ppl_infer_with_trainer.py \
    --model_name_or_path /home/feihm/llm-fei/ATM-RAG/fei-scripts/experiments/model_final/ \
    --input_file ${jsonl_root}/NQ_fab.jsonl \
    --per_device_eval_batch_size 32 \
    --output ${jsonl_root}/NQ_fab_score.csv