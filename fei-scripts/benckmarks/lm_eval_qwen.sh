# lm_eval --model hf \
#     --model_args pretrained=/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/ATM_RAG_0524/forEval \
#     --tasks triviaqa  \
#     --device cuda:0,cuda:1,cuda:2,cuda:3 \
#     --batch_size auto:8 \
#     --limit 1000


export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=2,3,4,5

lm_eval --model vllm \
    --model_args pretrained=/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/pre_exp/finetuned_model,tensor_parallel_size=4,dtype=auto   \
    --tasks rag_qa_advanced  \
    --write_out \
    --output_path /home/feihm/llm-fei/ATM-RAG/fei-scripts/benckmarks/result4sft_finetuned \
    --batch_size auto:8 \
    --limit 1000

# triviaqa
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
#     $(which lm_eval) \
#     --model vllm \
#     --model_args pretrained=unsloth/Qwen2.5-7B-Instruct,dtype=float16,enable_lora=True,lora_local_path=/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/ATM_RAG_0524/gen_model/model_mito_final/  \
#     --tasks triviaqa \
#     --write_out \
#     --log_samples \
#     --output_path /home/feihm/llm-fei/ATM-RAG/fei-scripts/benckmarks/results.json \
#     --batch_size auto:8 \
#     --limit 1000
