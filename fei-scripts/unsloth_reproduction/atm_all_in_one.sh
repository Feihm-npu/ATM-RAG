#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

## Variables

NUM_RUN=1
ORI_ADV_MODEL=unsloth/Qwen2.5-7B
ORI_GEN_MODEL=unsloth/Qwen2.5-7B

# DATASET_PATH=/home/feihm/llm-fei/Data/NQ/
DATASET_PATH=~/llm-fei/Data/NQ/contriever_nq/
DS=NQ
EXP_PATH=./ATM_RAG_0403_2
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

# Part 2: Merge FAB dataset (json)
if [ -f "${FAB_DS_PATH}${DS}.json" ]; then
    echo "Second part (FAB json) already exists, skipping."
else
    echo "Second part started!"

    python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_build_data/fab_merge.py \
        --ds_path ${DATASET_PATH}${DS}.json \
        --fab_path ${FAB_DS_PATH}${DS}.csv \
        --dest_dir ${FAB_DS_PATH}${DS}.json

    if [ ! -f "${FAB_DS_PATH}${DS}.json" ]; then
        echo "Error: FAB merge failed!"
        exit 1
    fi
    echo "Second part completed!"
fi

echo "Step 0 completed!"

########### Step 1: This script is used for initial fine-tuning the generator, only need to be run once.

echo "Step 1: Prepare dataset for SFT"

# Part 1: Prepare SFT dataset
if [ -f "$SFT_DS_PATH/dataset_info.json" ]; then
    echo "SFT dataset already exists, skipping data preparation."
else
    echo "Data preparing"
    python /home/feihm/llm-fei/ATM-RAG/atm_train/generator_sft/generator_sft_data_prepare.py \
        --model_path $ORI_GEN_MODEL \
        --data_path ${FAB_DS_PATH}${DS}.json \
        --dst_path $SFT_DS_PATH
            # --data_path ${DATASET_PATH}${DS}.json \
    if [ ! -f "$SFT_DS_PATH/dataset_info.json" ]; then
        echo "Error: SFT dataset preparation failed!"
        exit 1
    fi
    echo "Data prepared"
fi

# Part 2: Fine-tune the generator
if [ -f "${GEN_MODEL_PATH}adapter_config.json" ]; then
    echo "Generator model already exists, skipping fine-tuning."
else
    echo "Start finetuning the generator"
    python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_tuning_generator.py \
        --model_name_or_path $ORI_GEN_MODEL \
        --train_data $SFT_DS_PATH \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --bf16 \
        --output_dir $GEN_MODEL_PATH \
        --lr_scheduler_type cosine \
        --num_train_epochs 1 \
        --learning_rate 5e-6 \
        --wandb_project atm-sft-unsloth

    if [ ! -f "${GEN_MODEL_PATH}adapter_config.json" ]; then
        echo "Error: Generator fine-tuning failed!"
        exit 1
    fi
    echo "Generator SFT completed!"
fi

# Part 3: Merge model for vllm
if [ -f "${GEN_MODEL_VLLM_PATH}config.json" ]; then
    echo "vLLM model already exists, skipping merging."
else
    echo "Merging model for vllm"

    python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/merge_lora2vllm.py \
        --base_model_name $ORI_GEN_MODEL \
        --adapter_dir $GEN_MODEL_PATH \
        --output_dir $GEN_MODEL_VLLM_PATH

    if [ ! -f "${GEN_MODEL_VLLM_PATH}config.json" ]; then
        echo "Error: Model merging for vllm failed!"
        exit 1
    fi
fi

echo "Step 1 completed!"

########### Step 2: This script is used for building the DPO pairwise dataset, using the ppl output as the score.

echo "Step 2: Build DPO dataset"

# Part 1: Generate score for DPO dataset
if [ -f "${DPO_DS_PATH}${DS}_score.csv" ]; then
    echo "DPO score already exists, skipping score generation."
else
    echo "Generating score for DPO dataset"
    torchrun --nnodes=1 --nproc_per_node=$world_size \
        /home/feihm/llm-fei/ATM-RAG/atm_train/ppl_infer/ppl_infer_with_trainer.py \
        --model_name_or_path $GEN_MODEL_VLLM_PATH \
        --input_file ${FAB_DS_PATH}${DS}.json \
        --per_device_eval_batch_size 32 \
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

echo "Step 2 completed!"

########### Step 3: This script is used for training the attacker model with DPO.

echo "Step 3: DPO training"

# Part 1: Train the attacker model
if [ -f "${ADV_MODEL_PATH}adapter_config.json" ]; then
    echo "Attacker model already exists, skipping training."
else
    echo "Training started"
    python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s3_adversary_dpo.py \
      --model_name $ORI_GEN_MODEL \
      --train_file ${DPO_DS_PATH}${DS}_dpo.json \
      --output_dir $ADV_MODEL_PATH \
      --wandb_project unsloth-s3-attacker-dpo \
      --bf16

    if [ ! -f "${ADV_MODEL_PATH}adapter_config.json" ]; then
        echo "Error: DPO training failed!"
        exit 1
    fi
    echo "Training completed!"
fi

echo "Step 3 completed!"

########### Step 4: This script is used for generating fab dataset for MITO

echo "Step 4: MITO data preparing"

# Part 1: Generate MITO dataset
# if [ -f "${MITO_DS_PATH}${DS}.json" ]; then
#     echo "MITO dataset already exists, skipping generation."
# else
#     python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_build_data/feb_merge_updated.py \
#         --fab-path ${FAB_DS_PATH}${DS}.csv \
#         --ds-source ${DATASET_PATH}${DS}.json \
#         --dest-dir ${MITO_DS_PATH}${DS}.json

#     if [ ! -f "${MITO_DS_PATH}${DS}.json" ]; then
#         echo "Error: MITO data preparation failed!"
#         exit 1
#     fi
# fi

# Part 1: Merge adv model for vllm
if [ -f "${ADV_MODEL_VLLM_PATH}config.json" ]; then
    echo "vLLM model already exists, skipping merging."
else
    echo "Merging model for vllm"

    python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/merge_lora2vllm.py \
        --base_model_name $ORI_ADV_MODEL \
        --adapter_dir $ADV_MODEL_PATH \
        --output_dir $ADV_MODEL_VLLM_PATH

    if [ ! -f "${ADV_MODEL_VLLM_PATH}config.json" ]; then
        echo "Error: Model merging for vllm failed!"
        exit 1
    fi
fi

# Part 2: Generate initial MITO dataset (CSV)
if [ -f "${MITO_DS_PATH}${DS}.csv" ]; then
    echo "First part (DPO CSV) already exists, skipping."
else
    echo "First part started!"
    python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/sF0_generator_fake.py \
        --model_name $ADV_MODEL_VLLM_PATH \
        --world_size $world_size \
        --ds_name ${DATASET_PATH}${DS}.json \
        --dest_dir ${MITO_DS_PATH}${DS}.csv

    if [ -f "${MITO_DS_PATH}${DS}.csv" ]; then
        echo "Error: FAB CSV dataset generation failed!"
        exit 1
    fi
    echo "First part completed!"
fi

# Part 3: Merge FAB dataset (json) for MITO
if [ -f "${MITO_DS_PATH}${DS}.json" ]; then
    echo "Second part (FAB json) already exists, skipping."
else
    echo "Second part started!"

    python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_build_data/fab_merge.py \
        --ds_path ${DATASET_PATH}${DS}.json \
        --fab_path ${MITO_DS_PATH}${DS}.csv \
        --dest_dir ${MITO_DS_PATH}${DS}.json

    if [ ! -f "${MITO_DS_PATH}${DS}.json" ]; then
        echo "Error: FAB merge failed!"
        exit 1
    fi
    echo "Second part completed!"
fi

echo "Step 0 completed!"


echo "Step 4 completed!"

########### Step 5: This script is used for MITO training the generator.

echo "Step 5: MITO training the generator"

# Part 1: Train the generator with MITO
# if [ -d "$GEN_MODEL_PATH" ]; then
#     echo "Generator model already exists, skipping MITO training."
# else
python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/dpo_test.py \
    --model_name $GEN_MODEL_PATH \
    --dataset_name ${MITO_DS_PATH}${DS}.json \
    --batch_size 10 \
    --num_train_epochs 1 \
    --output_dir $GEN_MODEL_PATH

#     if [ ! -d "$GEN_MODEL_PATH" ]; then
#         echo "Error: MITO training failed!"
#         exit 1
#     fi
# fi
echo "Last step needs human check!"
echo "All steps completed successfully!"