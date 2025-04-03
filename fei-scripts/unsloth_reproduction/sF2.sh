## This script is used for build the DPO pairwise dataset, using the ppl output as the score.
## The model used is the generator after fine-tuning in step 1 (sF1.sh) or later after MITO training in step 4.


export CUDA_VISIBLE_DEVICES=4,5

jsonl_root=/home/feihm/llm-fei/Data/ATM/test_data_with_fabs/NQ


OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=4,0 torchrun --nnodes=1 --nproc_per_node=2 \
    /home/feihm/llm-fei/ATM-RAG/atm_train/ppl_infer/ppl_infer_with_trainer.py \
    --model_name_or_path /home/feihm/llm-fei/ATM-RAG/fei-scripts/experiments1/model_final/ \
    --input_file ${jsonl_root}/NQ_fab.jsonl \
    --per_device_eval_batch_size 32 \
    --output ${jsonl_root}/NQ_fab_score.csv


## Then use the score to build the DPO dataset.


python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_dpo/build_compare_dpo_data.py \
    --input_score ${jsonl_root}/NQ_fab_score.csv \
    --input_docs /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/ask_output/NQ/NQ_fab.csv \
    --ds_name 'dummy' \
    --output ${jsonl_root}/NQ_dpo.jsonl