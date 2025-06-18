export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=1,2

lm_eval --model vllm \
    --model_args pretrained=/home/feihm/llm-fei/ATM-RAG/fei-scripts/unsloth_reproduction/pre_exp/finetuned_model_0612/model_mito_final,tensor_parallel_size=2,dtype=auto   \
    --tasks triviaqa,rag_qa_advanced \
    --batch_size auto:8 \

    # --write_out \
    # --log_samples \
    # --output_path /home/feihm/llm-fei/ATM-RAG/fei-scripts/benckmarks/result4mito_0612 \

# lm_eval --model vllm \
#     --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,tensor_parallel_size=4,dtype=auto   \
#     --tasks rag_qa_advanced  \
#     --batch_size auto:8 \
#     --write_out \
#     --log_samples \
#     --output_path /home/feihm/llm-fei/ATM-RAG/fei-scripts/benckmarks/results_ORI_rag_qa \
#     --limit 1000

## Original Qwen: triviaqa 0.272, rag_qa 0.394
## SFT model: triviaqa 0.377, rag_qa 0.465
## MITO model: triviaqa 0.445, rag_qa 0.455



# triviaqa nq_open rag_qa_advanced



## mito
# |     Tasks     |Version|     Filter      |n-shot|  Metric   |   |Value|   |Stderr|
# |---------------|------:|-----------------|-----:|-----------|---|----:|---|-----:|
# |rag_qa_advanced|      1|remove_whitespace|     0|exact_match|↑  |0.455|±  |0.0158|
# |triviaqa       |      3|remove_whitespace|     0|exact_match|↑  |0.445|±  |0.0157|

## sft
# |     Tasks     |Version|     Filter      |n-shot|  Metric   |   |Value|   |Stderr|
# |---------------|------:|-----------------|-----:|-----------|---|----:|---|-----:|
# |rag_qa_advanced|      1|remove_whitespace|     0|exact_match|↑  |0.465|±  |0.0158|
# |triviaqa       |      3|remove_whitespace|     0|exact_match|↑  |0.377|±  |0.0153|


# |     Tasks     |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
# |---------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
# |rag_qa_advanced|      1|remove_whitespace|     0|exact_match|↑  |0.4925|±  |0.0112|
# |triviaqa       |      3|remove_whitespace|     0|exact_match|↑  |0.1746|±  |0.0028|