ORI_ADV_MODEL=unsloth/Qwen2.5-7B-Instruct
ORI_GEN_MODEL=unsloth/Qwen2.5-7B-Instruct

# DATASET_PATH=/home/feihm/llm-fei/Data/NQ/
DATASET_PATH=~/llm-fei/Data/NQ/contriever_nq_all_train/
#DATASET_PATH=/home/feihm/llm-fei/zeming/GraphRAG/Data/datasets/ATM
DS=train
EXP_PATH=./ATM_RAG_0516_all
ADV_MODEL_PATH=${EXP_PATH}/adv_model/
ADV_MODEL_VLLM_PATH=${EXP_PATH}/adv_model_vllm/
GEN_MODEL_PATH=${EXP_PATH}/gen_model/model_mito_final
GEN_MODEL_VLLM_PATH=${EXP_PATH}/gen_model_vllm/
GEN_MODEL_next_VLLM_PATH=${EXP_PATH}/gen_model_vllm_next/

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
mkdir -p $GEN_MODEL_next_VLLM_PATH
mkdir -p $FAB_DS_PATH
mkdir -p $SFT_DS_PATH
mkdir -p $DPO_DS_PATH
mkdir -p $MITO_DS_PATH

## Training environment
export CUDA_VISIBLE_DEVICES=0,4,5,6
export OMP_NUM_THREADS=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
world_size=4

echo "Merging model for vllm"

python /home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/merge_lora2vllm.py \
    --base_model_name $ORI_GEN_MODEL \
    --adapter_dir $GEN_MODEL_PATH \
    --output_dir $GEN_MODEL_VLLM_PATH
