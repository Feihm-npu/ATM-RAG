#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

## Variables

ORI_ADV_MODEL=Qwen/Qwen2.5-7B-Instruct
ORI_GEN_MODEL=Qwen/Qwen2.5-7B-Instruct

# DATASET_PATH=/home/feihm/llm-fei/Data/NQ/
DATASET_PATH=~/llm-fei/Data/NQ/contriever_nq_all_train/
#DATASET_PATH=/home/feihm/llm-fei/zeming/GraphRAG/Data/datasets/ATM
DS=train
EXP_PATH=./ATM_RAG_0603
PRE_EXP=./pre_exp

ADV_MODEL_PATH=${EXP_PATH}/adv_model/
ADV_MODEL_VLLM_PATH=${EXP_PATH}/adv_model_vllm/
GEN_MODEL_PATH=${PRE_EXP}/finetuned_model/
GEN_MODEL_VLLM_PATH=${EXP_PATH}/gen_model_vllm/

FAB_DS_PATH=${PRE_EXP}/fab_ds/
SFT_DS_PATH=${PRE_EXP}/fab_sft/
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export OMP_NUM_THREADS=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# This is added to solve the conflicts of the conflicts rooted from the combination: deepspeed + HF Transformers + PyTorch 2.2+

world_size=4
vllm_world_size=4
########### Step 0: Create initial fab dataset 

echo "Step 0: Create initial fab dataset"

# Part 1: Generate initial FAB dataset (CSV)
if [ -f "${FAB_DS_PATH}${DS}.csv" ]; then
    echo "First part (FAB CSV) already exists, skipping."
else
    echo "First part started!"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/sF0_generator_fake.py \
        --model_name $ORI_ADV_MODEL \
        --world_size $vllm_world_size \
        --max_new_tokens 512 \
        --num_proc 16 \
        --ds_name ${DATASET_PATH}${DS}.json \
        --dest_dir ${FAB_DS_PATH}${DS}.csv

    if [ ! -f "${FAB_DS_PATH}${DS}.csv" ]; then
        echo "Error: FAB dataset generation failed!"
        exit 1
    fi
    echo "First part completed!"
fi

# Part 2: Merge FAB dataset (json)
if [ -f "${FAB_DS_PATH}${DS}.json" ]; then
    echo "Second part (FAB json) already exists, skipping."
else
    echo "Second part started!"

    python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_build_data/fab_merge.py \
        --ds_path ${DATASET_PATH}${DS}.json \
        --num_proc 32 \
        --fab_path ${FAB_DS_PATH}${DS}.csv \
        --dest_dir ${FAB_DS_PATH}${DS}.json

    if [ ! -f "${FAB_DS_PATH}${DS}.json" ]; then
        echo "Error: FAB merge failed!"
        exit 1
    fi
    echo "Second part completed!"
fi

echo "Step 0 completed!"
echo "*********************************************************************************"

########### Step 1: This script is used for initial fine-tuning the generator, only need to be run once.

echo "Step 1: Prepare dataset for SFT"

# Part 1: Prepare SFT dataset
if [ -f "$SFT_DS_PATH/dataset_info.json" ]; then
    echo "SFT dataset already exists, skipping data preparation."
else
    echo "Data preparing"
    python /home/feihm/llm-fei/ATM-RAG/atm_train/generator_sft/generator_sft_data_prepare_qwen.py \
        --model_path $ORI_GEN_MODEL \
        --data_path ${DATASET_PATH}${DS}.json \
        --dst_path $SFT_DS_PATH
            # --data_path ${DATASET_PATH}${DS}.json \
    if [ ! -f "$SFT_DS_PATH/dataset_info.json" ]; then
        echo "Error: SFT dataset preparation failed!"
        exit 1
    fi
    echo "Data prepared"
fi

# Part 2: Fine-tune the generator
if [ -f "${GEN_MODEL_PATH}config.json" ]; then
    echo "Generator model already exists, skipping fine-tuning."
else
    echo "Start finetuning the generator"
    accelerate launch --config_file ds_configs/default_config.yaml /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_tuning_generator.py \
        --model_name_or_path $ORI_GEN_MODEL \
        --train_data $SFT_DS_PATH \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --max_steps 100 \
        --bf16 \
        --output_dir $GEN_MODEL_PATH \
        --lr_scheduler_type cosine \
        --num_train_epochs 1 \
        --learning_rate 5e-6 \
        --wandb_project atm-sft-qwen

    if [ ! -f "${GEN_MODEL_PATH}config.json" ]; then
        echo "Error: Generator fine-tuning failed!"
        exit 1
    fi
    echo "Generator SFT completed!"
fi

echo "*********************************************************************************"

########### Step 1: This script is used for building the DPO pairwise dataset, using the ppl output as the score.
echo "DPO start!"
echo "Step 1: Build DPO dataset"

# Part 1: Generate score for DPO dataset
if [ -f "${DPO_DS_PATH}${DS}_score.csv" ]; then
    echo "DPO score already exists, skipping score generation."
else
    echo "Generating score for DPO dataset"
    accelerate launch --config_file /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/ds_configs/ds_config_zero3_ppl.yaml \
        /home/feihm/llm-fei/ATM-RAG/atm_train/ppl_infer/ppl_infer_with_trainer_qwen.py \
        --model_name_or_path $GEN_MODEL_PATH \
        --input_file ${FAB_DS_PATH}${DS}.json \
        --per_device_eval_batch_size 80 \
        --dataloader_num_workers 8 \
        --num_procs 16 \
        --bf16 \
        --output ${DPO_DS_PATH}${DS}_score.csv

    if [ ! -f "${DPO_DS_PATH}${DS}_score.csv" ]; then
        echo "Error: DPO score generation failed!"
        exit 1
    fi
    echo "Score collected"
fi

# Part 2: Build pairwise data for DPO
if [ -f "${DPO_DS_PATH}${DS}_dpo.json" ]; then
    echo "DPO dataset already exists, skipping pairwise data generation."
else
    echo "Build pairwise data for DPO"
    python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_dpo/build_compare_dpo_data.py \
        --input_score ${DPO_DS_PATH}${DS}_score.csv \
        --input_docs ${FAB_DS_PATH}${DS}.csv \
        --ds_name ${FAB_DS_PATH}${DS}.json \
        --output ${DPO_DS_PATH}${DS}_dpo.json

    if [ ! -f "${DPO_DS_PATH}${DS}_dpo.json" ]; then
        echo "Error: DPO dataset generation failed!"
        exit 1
    fi
    echo "DPO dataset generated"
fi

# echo "Step 2 completed!"

########### Step 3: This script is used for training the attacker model with DPO.

echo "Step 3: DPO training"

# Part 3: Train the attacker model
if [ -f "${ADV_MODEL_PATH}config.json" ]; then
    echo "Attacker model already exists, skipping training."
else
    echo "Training started"
    accelerate launch --config_file ds_configs/default_config.yaml /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s3_adversary_dpo.py \
      --model_name $ORI_GEN_MODEL \
      --ref_model $ORI_GEN_MODEL \
      --train_file ${DPO_DS_PATH}${DS}_dpo.json \
      --per_device_train_batch_size 2 \
      --max_steps 1 \
      --gradient_accumulation_steps 1 \
      --output_dir $ADV_MODEL_PATH \
      --wandb_project unsloth-s3-attacker-dpo \
      --bf16

    if [ ! -f "${ADV_MODEL_PATH}config.json" ]; then
        echo "Error: DPO training failed!"
        exit 1
    fi
    echo "Training completed!"
fi

echo "Step 3 completed!"

########### Step 4: This script is used for generating fab dataset for MITO

echo "Step 4: MITO data preparing"


# Part 2: Generate initial MITO dataset (CSV)
if [ -f "${MITO_DS_PATH}${DS}.csv" ]; then
    echo "First part (DPO CSV) already exists, skipping."
else
    echo "First part started!"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/sF0_generator_fake_opt.py \
        --model_name $ADV_MODEL_PATH \
        --world_size $world_size \
        --max_new_tokens 512 \
        --num_proc 16 \
        --ds_name ${DATASET_PATH}${DS}.json \
        --dest_dir ${MITO_DS_PATH}${DS}.csv

    if [ ! -f "${MITO_DS_PATH}${DS}.csv" ]; then
        echo "Error: FAB CSV dataset generation failed!"
        exit 1
    fi
    echo "First part completed!"
fi

# Part 3: Merge FAB dataset (json) for MITO, but now the data looks like:
# {'question': Value(dtype='string', id=None),
#  'answers': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
#  'ctxs': [{'hasanswer': Value(dtype='bool', id=None),
#    'id': Value(dtype='string', id=None),
#    'score': Value(dtype='string', id=None),
#    'text': Value(dtype='string', id=None),
#    'title': Value(dtype='string', id=None)}]}
if [ -f "${MITO_DS_PATH}${DS}.json" ]; then
    echo "Second part (FAB json) already exists, skipping."
else
    echo "Second part started!"

    python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_build_data/fab_merge.py \
        --ds_path ${DATASET_PATH}${DS}.json \
        --fab_path ${MITO_DS_PATH}${DS}.csv \
        --num_procs 32 \
        --dest_dir ${MITO_DS_PATH}${DS}.json

    if [ ! -f "${MITO_DS_PATH}${DS}.json" ]; then
        echo "Error: FAB merge failed!"
        exit 1
    fi
    echo "Second part completed!"
fi

# Part 4: Generate MITO dataset, the output should have features {'prompt','chosen', 'rejected'}
if [ -f "${MITO_DS_PATH}${DS}_mito.json" ]; then
    echo "MITO dataset already exists, skipping generation."
else
    python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/sF4_build_mito.py \
        --ds-source ${MITO_DS_PATH}${DS}.json \
        --dest-dir ${MITO_DS_PATH}${DS}_mito.json

    if [ ! -f "${MITO_DS_PATH}${DS}_mito.json" ]; then
        echo "Error: MITO data preparation failed!"
        exit 1
    fi
fi


echo "Step 4 completed!"

########### Step 5: This script is used for MITO training the generator.

echo "Step 5: MITO training the generator"

# Part 1: Train the generator with MITO
# if [ -d "$GEN_MODEL_PATH" ]; then
#     echo "Generator model already exists, skipping MITO training."
# else


# CUDA_VISIBLE_DEVICES=4 python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/dpo_test.py \
#     --model_name $GEN_MODEL_PATH \
#     --dataset_name ${MITO_DS_PATH}${DS}_mito.json \
#     --batch_size 2 \
#     --num_train_epochs 1 \
#     --output_dir $GEN_MODEL_PATH 

# multiple gpus CUDA_VISIBLE_DEVICES=0 python mito_ds.py \

# CUDA_VISIBLE_DEVICES=0 python mito_ds.py \
accelerate launch --config_file ds_configs/default_config.yaml mito_ds.py \
    --model_name $GEN_MODEL_PATH \
    --dataset_name ${MITO_DS_PATH}${DS}_mito.json  \
    --batch_size 1 \
    --max_steps 10 \
    --output_dir $GEN_MODEL_PATH  


    # --max_steps 100 \

#     if [ ! -d "$GEN_MODEL_PATH" ]; then
#         echo "Error: MITO training failed!"
#         exit 1
#     fi
# fi
echo "Last step needs mannual check!"
echo "All steps completed successfully!"