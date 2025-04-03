from datasets import load_dataset, Dataset  # 导入 Hugging Face datasets 库，用于加载和处理数据集
from pathlib import Path  # 用于文件路径操作
import pandas as pd  # 导入 pandas 处理 CSV 数据
import argparse  # 用于解析命令行参数

# 预处理替换规则，将部分标记标准化
pre_replace_pairs = [
    ('TEXT', '<text>'),  # 统一文本标记
    ('Text:', '<text>'),  
    ('Text', '<text>'),  
    ('TITLE', '<title>'),  # 统一标题标记
    ('Title:', '<title>'),  
    ('Title', '<title>'),  
    ('##', '#'),   # 规范 markdown 标题格式
]

# 后处理替换规则，去除无意义字符
post_replace_pairs = [
    ('<text>', ''),
    ('<title>', ''),   
    ('\n', ''),  # 移除换行符
    (':', ''),  # 去掉冒号
]

# 格式化分割函数，解析文本并提取标题和正文
def format_split(output):
    seps = ['#', '<text>']  # 可能的分隔符
    output = pre_replace_seps(output)  # 预处理替换
    
    out = dict(title="", text="")  # 初始化存储结果
    for sep in seps:
        if sep not in output:
            continue
        splitted = output.split(sep)  # 按照分隔符拆分
        if len(splitted) != 2:
            continue
        p_splitted = [piece for piece in splitted if piece]  # 过滤掉空字符串
        if len(p_splitted) != 2:
            continue
        else:
            out['title'] = p_splitted[0].strip()  # 去掉两端空格
            out['text'] = p_splitted[1].strip()
    
    if out['title'] == "" and out['text'] == "":
        out['text'] = output  # 若未能提取标题，则认为整个文本都是正文
    
    out = {k: post_replace(v).strip() for k, v in out.items()}  # 应用后处理替换
    return out

# 提取文本特征
def extract_feat(example):
    return format_split(example['output'])  # 解析 'output' 字段

# 后处理替换函数
def post_replace(item):
    for one_replace_pair in post_replace_pairs:
        item = item.replace(*one_replace_pair)
    return item

# 预处理替换函数
def pre_replace_seps(output):
    for one_replace_pair in pre_replace_pairs:
        output = output.replace(*one_replace_pair)
    return output

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--ds_path", default='nq-test', type=str)  # 数据集名称
    parser.add_argument("--fab_path", default='NQ_fab', type=str)  # FAB 文件路径
    parser.add_argument("--num_dups", default=5, type=int)  # 重复次数
    parser.add_argument("--epoch_suffix", default=0, type=int)  # 训练轮次后缀
    parser.add_argument("--dest_dir", required=True, type=str)  # 输出目录
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()  # 解析参数

    ds_path = args.ds_path  # 获取数据集名称
    fab_path = args.fab_path
    dest_path = args.dest_dir  # 输出目录

    fab_file_path = f'{fab_path}'  # CSV 数据路径
    ds_source_path = f'{ds_path}'  # 原始 JSONL 数据集路径
    
    # dest_path = Path(f'/home/feihm/llm-fei/Data/ATM/test_data_with_fabs')  # 目标路径
    # if not dest_path.exists():
    #     dest_path.mkdir()  # 如果路径不存在，则创建
    
    fab_df = pd.read_csv(fab_file_path)  # 读取 CSV 文件
    fab_df = fab_df.fillna(fab_df['output_0'].iloc[0])  # 处理缺失值，填充为第一个 'output_0' 的值

    rds = load_dataset('json', data_files=ds_source_path, split='train')  # 加载 JSON 数据集
    
    nads = []  # 存储新的数据集
    for idx in range(args.num_dups):
        outputs = fab_df[f'output_{idx}'].astype(str).tolist()  # 获取当前 'output' 列数据
        ads = Dataset.from_dict({'output': outputs})  # 转换为 Dataset 格式
        nads.append(ads.map(extract_feat, num_proc=8, remove_columns=ads.column_names))  # 并行处理并去掉旧列
    
    rds = rds.to_list()  # 转换为列表格式
    
    # 遍历数据集，为每个样本插入新的上下文信息
    for idx, item in enumerate(rds):
        for jdx in range(args.num_dups):
            insert_item = {
                "id": f"fab_{args.epoch_suffix}_q{idx}_d{jdx}",  # 生成唯一 ID
                "title": nads[jdx][idx]['title'],  # 处理后的标题
                "text": nads[jdx][idx]['text'],  # 处理后的正文
                "score": '2',  # 固定评分
                'hasanswer': True,  # 标记为有答案
            }
            rds[idx]['ctxs'].insert(0, insert_item)  # 插入新内容
    
    rds = Dataset.from_list(rds)  # 转换回 Dataset
    rds.to_json(f'{dest_path}')  # 保存为 JSONL 文件
