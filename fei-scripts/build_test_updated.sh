#!/bin/bash

python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_build_data/feb_merge_updated.py \
    --ds_name NQ/NQ \
    --dest_dir /home/feihm/llm-fei/Data/ATM/test_data_with_fabs_updated

echo "Second part completed!"


# python generator_sft_data_prepare.py