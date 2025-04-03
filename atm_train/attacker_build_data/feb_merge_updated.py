from datasets import load_dataset, Dataset
from pathlib import Path
import pandas as pd
import argparse

# 替换规则
pre_replace_pairs = [
    ('TEXT', '<text>'), ('Text:', '<text>'), ('Text', '<text>'),
    ('TITLE', '<title>'), ('Title:', '<title>'), ('Title', '<title>'),
    ('##', '#'),
]

post_replace_pairs = [
    ('<text>', ''), ('<title>', ''), ('\n', ''), (':', '')
]

def format_split(output):
    seps = ['#', '<text>']
    output = pre_replace_seps(output)
    out = dict(title="", text="")
    for sep in seps:
        if sep not in output:
            continue
        splitted = output.split(sep)
        if len(splitted) != 2:
            continue
        p_splitted = [piece for piece in splitted if piece]
        if len(p_splitted) != 2:
            continue
        out['title'] = p_splitted[0].strip()
        out['text'] = p_splitted[1].strip()
    if out['title'] == "" and out['text'] == "":
        out['text'] = output
    out = {k: post_replace(v).strip() for k, v in out.items()}
    return out

def extract_feat(example):
    return format_split(example['output'])

def post_replace(item):
    for pair in post_replace_pairs:
        item = item.replace(*pair)
    return item

def pre_replace_seps(output):
    for pair in pre_replace_pairs:
        output = output.replace(*pair)
    return output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fab-path", default='nq-test', type=str)
    parser.add_argument("--num_dups", default=5, type=int)
    parser.add_argument("--epoch_suffix", default=0, type=int)
    parser.add_argument("--dest-dir", required=True, type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # ds_name = args.ds_name
    fab_file_path = f'{args.fab_path}'
    ds_source_path = f'{args.dest_dir}'

    dest_path = Path(args.dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    fab_df = pd.read_csv(fab_file_path).fillna(method='ffill')

    rds = load_dataset('json', data_files=ds_source_path, split='train')

    nads = []
    for idx in range(args.num_dups):
        outputs = fab_df[f'output_{idx}'].tolist()
        ads = Dataset.from_dict({'output': outputs})
        nads.append(ads.map(extract_feat, num_proc=8, remove_columns=ads.column_names))

    rds = rds.to_list()

    for idx, item in enumerate(rds):
        question = item['question']
        answer = item['answers'][0] if isinstance(item['answers'], list) else item['answers']
        raw_prompt = f"Question: {question}\nContext:\n"

        clean_ctxs = item['ctxs'][args.num_dups:]
        fab_ctxs = []

        # 插入前 num_dups 个 fabrication 文档
        for jdx in range(args.num_dups):
            fab = {
                "id": f"fab_{args.epoch_suffix}_q{idx}_d{jdx}",
                "title": nads[jdx][idx]['title'],
                "text": nads[jdx][idx]['text'],
                "score": '2',
                "hasanswer": True,
            }
            item['ctxs'].insert(0, fab)
            fab_ctxs.append(fab)

        def ctx_to_str(ctx):
            return f"TITLE: {ctx['title']}\nTEXT: {ctx['text']}"

        # 构造 prompt 与 adv_prompt
        clean_texts = [ctx_to_str(ctx) for ctx in clean_ctxs]
        fab_texts = [ctx_to_str(ctx) for ctx in fab_ctxs]

        prompt = raw_prompt + "\n\n".join(clean_texts)
        adv_prompt = raw_prompt + "\n\n".join(fab_texts + clean_texts)

        item['prompt'] = prompt
        item['adv_prompt'] = adv_prompt
        item['answer'] = answer

    final_dataset = Dataset.from_list(rds)
    output_path = dest_path
    final_dataset.to_json(str(output_path))

    print(f"✅ Saved MITO-formatted dataset to: {output_path}")
