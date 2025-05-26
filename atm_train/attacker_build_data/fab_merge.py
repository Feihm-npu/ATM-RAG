from datasets import load_dataset, Dataset
from pathlib import Path
import pandas as pd
import argparse

# pre_replace_pairs = [
#     ('TEXT', '<text>'),
#     ('Text:', '<text>'),  
#     ('Text', '<text>'),  
#     ('TITLE', '<title>'),
#     ('Title:', '<title>'),  
#     ('Title', '<title>'),  
#     ('##', '#'),
# ]

pre_replace_pairs = [
    ('\xa0', ' '),  # 替换 non-breaking space 为普通空格（常见异常字符）
]


post_replace_pairs = [
    # ('<text>', ''),
    # ('<title>', ''),   
    ('\n', ''),
    # (':', ''),
]

def format_split(output):
    title, text = "", ""
    try:
        if '# Title' in output and '## Text' in output:
            parts = output.split('## Text')
            title_block = parts[0].replace('# Title', '').strip()
            text_block = parts[1].strip()
            title = post_replace(title_block)
            text = post_replace(text_block)
        else:
            text = post_replace(output.strip())
    except Exception as e:
        print(f"[!] Failed to split output: {e}")
        text = post_replace(output.strip())
    return {'title': title, 'text': text}


def extract_feat(example):
    return format_split(example['output'])

def post_replace(item):
    for one_replace_pair in post_replace_pairs:
        item = item.replace(*one_replace_pair)
    return item

def pre_replace_seps(output):
    for one_replace_pair in pre_replace_pairs:
        output = output.replace(*one_replace_pair)
    return output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", default='nq-train', type=str)
    parser.add_argument("--fab_path", default='generated_outputs.csv', type=str)
    parser.add_argument("--num_dups", default=5, type=int)
    parser.add_argument("--epoch_suffix", default=0, type=int)
    parser.add_argument("--dest_dir", required=True, type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    fab_df = pd.read_csv(args.fab_path)
    fab_df = fab_df.fillna(fab_df['output_0'].iloc[0])
    # rds = load_dataset('json', data_files=args.ds_path, split='train').select(range(100))
    rds = load_dataset('json', data_files=args.ds_path, split='train')
    nads = []

    for idx in range(args.num_dups):
        outputs = fab_df[f'output_{idx}'].astype(str).tolist()
        ads = Dataset.from_dict({'output': outputs})
        nads.append(ads.map(extract_feat, num_proc=8, remove_columns=ads.column_names))

    rds = rds.to_list()

    for idx, item in enumerate(rds):
        for jdx in range(args.num_dups):
            insert_item = {
                "id": f"fab_{args.epoch_suffix}_q{idx}_d{jdx}",
                "title": nads[jdx][idx]['title'],
                "text": nads[jdx][idx]['text'],
                "score": '2',
                'hasanswer': True,
            }
            rds[idx].setdefault('ctxs', []).insert(0, insert_item)

    rds = Dataset.from_list(rds)
    Path(args.dest_dir).parent.mkdir(parents=True, exist_ok=True)
    rds.to_json(args.dest_dir)  # expects a filename not a directory
