# epoch_suffix=${1}
# ds_name=${2}


jsonl_root=/home/feihm/llm-fei/Data/ATM/test_data_with_fabs/NQ



python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_dpo/build_compare_dpo_data.py \
    --input_score ${jsonl_root}/NQ_fab_score.csv \
    --input_docs /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/ask_output/NQ/NQ_fab.csv \
    --ds_name 'dummy' \
    --output /home/feihm/llm-fei/Data/ATM/test_data_with_fabs/NQ/NQ_dpo.jsonl
