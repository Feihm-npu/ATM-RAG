import pandas as pd
from datasets import load_dataset, Dataset, Features, Value
import argparse
import numpy as np
from pathlib import Path
from prompting_for_rag import get_prompt

example_format = 'TITLE {title} # TEXT {text}'
NUM_DUPS = 5

def format_row(example):
    item = {}
    try:
        item['example'] = example_format.format_map(example['passages']['passage_text'][0])
    except:
        item['example'] = example_format.format_map({
            "title": "<title>",
            "text": "<text>",
        })

    item['question'] = example['question']
    item['answers'] = example['answers']

    return {'prompt': get_prompt('atm_data_attacker', item)}

def get_minmax(scores):
    assert scores.shape[1] == NUM_DUPS
    
    max_value = np.argmax(scores, axis=-1)
    min_value = np.argmin(scores, axis=-1)
    
    return {
        "max": max_value,
        "min": min_value,
    }
    
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--input_score", type=str)
    parser.add_argument("--input_docs", type=str)

    parser.add_argument("--ds_name", default='nq-train', type=str)

    parser.add_argument("--output", required=True, type=str)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    ds_name = args.ds_name
    
    ds = load_dataset('json', data_files=ds_name, split='train')

    ds = ds.map(format_row, num_proc=8, remove_columns=ds.column_names)
    

    # Read and clean score CSV, replacing None/NaN with 2
    score_df = pd.read_csv(args.input_score)
    score_df = score_df.fillna(2)  # Replace NaN with 2
    score_df = score_df.replace([None, 'None', np.nan], 2)  # Replace None or 'None' strings with 2
    sdf = score_df.values.astype(float)  # Convert to float for argmax/argmin

    # Read docs CSV as strings
    docs_df = pd.read_csv(args.input_docs).astype(str)  # Ensure all are strings
    tdf = docs_df.values

    min_max = get_minmax(sdf)
    
    # Ensure chosen and rejected are strings
    chosen = tdf[np.arange(tdf.shape[0]), min_max['max']].tolist()
    rejected = tdf[np.arange(tdf.shape[0]), min_max['min']].tolist()

    # Convert to strings explicitly
    chosen = [str(x) for x in chosen]
    rejected = [str(x) for x in rejected]

    # Convert dataset to dict
    ds_dict = ds.to_dict()

    # Add chosen and rejected
    ds_dict['chosen'] = chosen
    ds_dict['rejected'] = rejected

    # Define features to ensure all fields are strings
    features = Features({
        'prompt': Value('string'),  # Existing field
        'chosen': Value('string'),  # New field
        'rejected': Value('string') # New field
    })

    # Create dataset with explicit features
    ds = Dataset.from_dict(ds_dict, features=features)
    
    # Save to JSON
    ds.to_json(args.output)