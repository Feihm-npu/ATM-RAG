#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

## Variables

ORI_ADV_MODEL=Qwen/Qwen2-0.5B-Instruct
ORI_GEN_MODEL=Qwen/Qwen2-0.5B-Instruct

# DATASET_PATH=/home/feihm/llm-fei/Data/NQ/
DATASET_PATH=~/llm-fei/Data/NQ/contriever_nq/
DS=NQ
EXP_PATH=./ATM_RAG_0421_only4eval
ADV_MODEL_PATH=${EXP_PATH}/adv_model/
ADV_MODEL_VLLM_PATH=${EXP_PATH}/adv_model_vllm/
GEN_MODEL_PATH=${EXP_PATH}/gen_model/
GEN_MODEL_VLLM_PATH=${EXP_PATH}/gen_model_vllm/
EVAL_MODEL_PATH=${EXP_PATH}/evaluations/

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
mkdir -p $EVAL_MODEL_PATH

## Training environment
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
world_size=2

########### Step 0: Create initial fab dataset 

echo "Evaluation"

# Part 1: Generate initial FAB dataset (CSV)
if [ -f "${EVAL_MODEL_PATH}${DS}_eval.csv" ]; then
    echo "First part (FAB CSV) already exists, skipping generation."
    python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/e2eval.py \
        --dataset ${DATASET_PATH}${DS}.json \
        --prediction ${EVAL_MODEL_PATH}${DS}_eval.csv \
        --num_dups 5
else
    echo "First part started!"
    python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/e1generate.py \
        --model_name $ORI_ADV_MODEL \
        --world_size $world_size \
        --ds_name ${DATASET_PATH}${DS}.json \
        --dest_dir ${EVAL_MODEL_PATH}${DS}_eval.csv


    if [ ! -f "${EVAL_MODEL_PATH}${DS}_eval.csv" ]; then
        echo "Error: evaluation failed!"
        exit 1
    else
        python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/e2eval.py \
        --dataset ${DATASET_PATH}${DS}.json \
        --prediction ${EVAL_MODEL_PATH}${DS}_eval.csv \
        --num_dups 5
    fi
    
    echo "First part completed!"
fi