#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

## Variables


ORI_ADV_MODEL=Qwen/Qwen2.5-7B-Instruct
ORI_GEN_MODEL=Qwen/Qwen2.5-7B-Instruct

# DATASET_PATH=/home/feihm/llm-fei/Data/NQ/
DATASET_PATH=~/llm-fei/Data/NQ/contriever_nq_all_train/

DS=train
EXP_PATH=./ATM_RAG_0612
PRE_EXP=./pre_exp


FT_MODEL_PATH=${PRE_EXP}/finetuned_model/
FAB_DS_PATH=${PRE_EXP}/fab_ds/
SFT_DS_PATH=${PRE_EXP}/fab_sft/


# Create directories if they don't exist
mkdir -p $EXP_PATH
mkdir -p $PRE_EXP
mkdir -p $FT_MODEL_PATH

#-----------------------------------
mkdir -p $FAB_DS_PATH
mkdir -p $SFT_DS_PATH


## Training environment
export CUDA_VISIBLE_DEVICES=0,3,4,5,6,7
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
if [ -f "${FT_MODEL_PATH}config.json" ]; then
    echo "Generator model already exists, skipping fine-tuning."
else
    echo "Start finetuning the generator"
    accelerate launch --config_file ds_configs/default_config.yaml /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s1_tuning_generator.py \
        --model_name_or_path $ORI_GEN_MODEL \
        --train_data $SFT_DS_PATH \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --bf16 \
        --output_dir $FT_MODEL_PATH \
        --lr_scheduler_type cosine \
        --num_train_epochs 1 \
        --max_steps 500 \
        --learning_rate 5e-6 \
        --wandb_project atm-sft-qwen

    if [ ! -f "${FT_MODEL_PATH}config.json" ]; then
        echo "Error: Generator fine-tuning failed!"
        exit 1
    fi
    echo "Generator SFT completed!"
fi

echo "*********************************************************************************"
echo "*********************************************************************************"
echo "************************** MITO Iteration ***************************************"
echo "*********************************************************************************"
echo "*********************************************************************************"

TOTAL_EPOCHS=2
CURRENT_EPOCH=0

for ((i=0; i<TOTAL_EPOCHS; i++)); do
    echo "EPOCH ${i}/${TOTAL_EPOCHS} starts"

    if [ $i -eq 0 ]; then
        echo "This is the first epoch, using the SFT model as the initial model, using the original adversarial model."
        FT_MODEL_PATH=${FT_MODEL_PATH}
        ORI_ADV_MODEL=${ORI_ADV_MODEL}
        FAB_DS_PATH=${FAB_DS_PATH}
    else
        echo "Using the model from the previous epoch."
        FT_MODEL_PATH=${EXP_PATH}/epoch_$((i-1))/gen_model/model_mito_final/
        ORI_ADV_MODEL=${EXP_PATH}/epoch_$((i-1))/adv_model/
        FAB_DS_PATH=${MITO_DS_PATH}
        # FT_MODEL_PATH: the path of the model for generating ppl and continue for mito training.
        # ORI_ADV_MODEL: the path of the original adversarial model, which is used for generating the DPO dataset. 
        # After the first epoch, it will be the model from the previous epoch.
        # FAB_DS_PATH: the path of the FAB dataset, which is used for generating the DPO dataset. Will update in each epoch.
    fi
    EPOCH_PATH=${EXP_PATH}/epoch_${i}
    DPO_DS_PATH=${EPOCH_PATH}/fab_dpo/
    MITO_DS_PATH=${EPOCH_PATH}/fab_mito/
    # MITO_MODEL_PATH=${EPOCH_PATH}/mito_model/
    GEN_MODEL_PATH=${EPOCH_PATH}/gen_model/
    ADV_MODEL_PATH=${EPOCH_PATH}/adv_model/


    mkdir -p $EPOCH_PATH $DPO_DS_PATH $MITO_DS_PATH $GEN_MODEL_PATH

    ########### Step 2: DPO Dataset Generation ###########
    echo "Step 2: DPO score generation"
    if [ -f "${DPO_DS_PATH}${DS}_score.csv" ]; then
        echo "DPO score already exists. Skipping generation."
    else
        accelerate launch --config_file /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/ds_configs/ds_config_zero3_ppl.yaml \
            /home/feihm/llm-fei/ATM-RAG/atm_train/ppl_infer/ppl_infer_with_trainer_qwen.py \
            --model_name_or_path $FT_MODEL_PATH \
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
    fi

    echo "Step 2: DPO dataset construction"
    if [ -f "${DPO_DS_PATH}${DS}_dpo.json" ]; then
        echo "DPO dataset already exists. Skipping generation."
    else
        python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_dpo/build_compare_dpo_data.py \
            --input_score ${DPO_DS_PATH}${DS}_score.csv \
            --input_docs ${FAB_DS_PATH}${DS}.csv \
            --ds_name ${FAB_DS_PATH}${DS}.json \
            --output ${DPO_DS_PATH}${DS}_dpo.json

        if [ ! -f "${DPO_DS_PATH}${DS}_dpo.json" ]; then
            echo "Error: DPO dataset generation failed!"
            exit 1
        fi
    fi

    ########### Step 3: DPO Training ###########
    echo "Step 3: Adversarial DPO training"
    if [ -f "${ADV_MODEL_PATH}config.json" ]; then
        echo "Attacker model already exists. Skipping training."
    else
        accelerate launch --config_file ds_configs/default_config.yaml \
            /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/s3_adversary_dpo.py \
            --model_name $ORI_ADV_MODEL \
            --ref_model $ORI_ADV_MODEL \
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
    fi

    ########### Step 4: MITO Dataset Preparation ###########
    echo "Step 4: MITO dataset preparation"
    if [ -f "${MITO_DS_PATH}${DS}.csv" ]; then
        echo "MITO CSV already exists. Skipping generation."
    else
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
    fi

    if [ -f "${MITO_DS_PATH}${DS}.json" ]; then
        echo "MITO merged JSON already exists. Skipping."
    else
        python /home/feihm/llm-fei/ATM-RAG/atm_train/attacker_build_data/fab_merge.py \
            --ds_path ${DATASET_PATH}${DS}.json \
            --fab_path ${MITO_DS_PATH}${DS}.csv \
            --num_procs 32 \
            --dest_dir ${MITO_DS_PATH}${DS}.json

        if [ ! -f "${MITO_DS_PATH}${DS}.json" ]; then
            echo "Error: FAB merge failed!"
            exit 1
        fi
    fi

    if [ -f "${MITO_DS_PATH}${DS}_mito.json" ]; then
        echo "MITO dataset already exists. Skipping."
    else
        python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/sF4_build_mito.py \
            --ds-source ${MITO_DS_PATH}${DS}.json \
            --dest-dir ${MITO_DS_PATH}${DS}_mito.json

        if [ ! -f "${MITO_DS_PATH}${DS}_mito.json" ]; then
            echo "Error: MITO data preparation failed!"
            exit 1
        fi
    fi

    ########### Step 5: MITO Training ###########
    echo "Step 5: MITO Training"
    if [ -f "${GEN_MODEL_PATH}model_mito_final/config.json" ]; then
        echo "Generator model already exists. Skipping MITO training."
    else
        accelerate launch --config_file ds_configs/default_config.yaml mito_ds.py \
            --model_name $FT_MODEL_PATH \
            --dataset_name ${MITO_DS_PATH}${DS}_mito.json \
            --batch_size 1 \
            --max_steps 10 \
            --output_dir $GEN_MODEL_PATH

        if [ ! -f "${GEN_MODEL_PATH}model_mito_final/config.json" ]; then
            echo "Error: MITO training failed!"
            exit 1
        fi
    fi

    echo "Epoch $i completed."
    echo "-----------------------------------------------------"
done

echo "EPOCH ${CURRENT_EPOCH+1}/${TOTAL_EPOCHS} completed successfully!"
