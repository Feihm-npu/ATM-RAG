

python /home/feihm/llm-fei/contriever/passage_retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages ~/llm-fei/Data/psgs_w100.tsv \
    --passages_embeddings "/home/feihm/llm-fei/Data/NQ/contriever-msmarco/wikipedia_embeddings/*" \
    --data ~/llm-fei/Data/NQ/train.json \
    --output_dir ~/llm-fei/Data/NQ/contriever_nq_all_train_msmarco

# python /home/feihm/llm-fei/Data/json2jsonl.py /home/feihm/llm-fei/Data/NQ/contriever_nq_all_train/test.json


# python /home/feihm/llm-fei/contriever/passage_retrieval.py \
#     --model_name_or_path facebook/contriever \
#     --passages ~/llm-fei/Data/psgs_w100.tsv \
#     --passages_embeddings "/home/feihm/llm-fei/Data/NQ/wikipedia_embeddings/*" \
#     --data ~/llm-fei/Data/NQ/train.json \
#     --output_dir ~/llm-fei/Data/NQ/contriever_nq_all_train

# python /home/feihm/llm-fei/Data/json2jsonl.py /home/feihm/llm-fei/Data/NQ/contriever_nq_all_train/test.json