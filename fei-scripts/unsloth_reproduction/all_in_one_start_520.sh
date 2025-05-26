#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

## Variables

ORI_ADV_MODEL=unsloth/Qwen2.5-7B-Instruct
ORI_GEN_MODEL=unsloth/Qwen2.5-7B-Instruct

# DATASET_PATH=/home/feihm/llm-fei/Data/NQ/
DATASET_PATH=~/llm-fei/Data/NQ/contriever_nq_all_train/
#DATASET_PATH=/home/feihm/llm-fei/zeming/GraphRAG/Data/datasets/ATM
DS=train
EXP_PATH=./ATM_RAG_0520_test
ADV_MODEL_PATH=${EXP_PATH}/adv_model/
ADV_MODEL_VLLM_PATH=${EXP_PATH}/adv_model_vllm/
GEN_MODEL_PATH=${EXP_PATH}/gen_model/
GEN_MODEL_VLLM_PATH=${EXP_PATH}/gen_model_vllm/

FAB_DS_PATH=${EXP_PATH}/fab_ds/
SFT_DS_PATH=${EXP_PATH}/fab_sft/
DPO_DS_PATH=${EXP_PATH}/fab_dpo/
MITO_DS_PATH=${EXP_PATH}/fab_mito/

# Create directories if they don't exist
mkdir -p $EXP_PATH
mkdir -p $ADV_MODEL_PATH
mkdir -p $ADV_MODEL_VLLM_PATH
mkdir -p $GEN_MODEL_PATH
mkdir -p $GEN_MODEL_VLLM_PATH
mkdir -p $FAB_DS_PATH
mkdir -p $SFT_DS_PATH
mkdir -p $DPO_DS_PATH
mkdir -p $MITO_DS_PATH

## Training environment
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
world_size=4

########### Step 0: Create initial fab dataset 

echo "Step 0: Create initial fab dataset"

# Part 1: Generate initial FAB dataset (CSV)
if [ -f "${FAB_DS_PATH}${DS}.csv" ]; then
    echo "First part (FAB CSV) already exists, skipping."
else
    echo "First part started!"
    python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/sF0_generator_fake.py \
        --model_name $ORI_ADV_MODEL \
        --world_size $world_size \
        --ds_name ${DATASET_PATH}${DS}.json \
        --dest_dir ${FAB_DS_PATH}${DS}.csv

    if [ ! -f "${FAB_DS_PATH}${DS}.csv" ]; then
        echo "Error: FAB dataset generation failed!"
        exit 1
    fi
    echo "First part completed!"
fi
