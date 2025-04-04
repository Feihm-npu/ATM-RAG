

python /home/feihm/llm-fei/contriever/passage_retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages ~/llm-fei/Data/psgs_w100.tsv \
    --passages_embeddings ~/llm-fei/Data/NQ/wikipedia_embeddings/passages_00 \
    --data ~/llm-fei/Data/NQ/test.json \
    --output_dir ~/llm-fei/Data/NQ/contriever_nq

python /home/feihm/llm-fei/Data/json2jsonl.py ~/llm-fei/Data/NQ/contriever_nq/test.json