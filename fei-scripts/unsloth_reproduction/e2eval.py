import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm

# 加载 HuggingFace 的评估指标
squad_metric = load("squad")

def normalize_text(text):
    return text.strip().lower()

def compute_subspan_em(pred, answers):
    """
    Subspan EM: 如果预测是参考答案的子串，或反之，也认为是匹配。
    """
    pred_norm = normalize_text(pred)
    answers_norm = [normalize_text(ans) for ans in answers]
    for ans in answers_norm:
        if pred_norm in ans or ans in pred_norm:
            return 1.0
    return 0.0

def evaluate(preds_df, dataset, output_keys):
    total = len(preds_df)
    metrics = {key: {"EM": [], "F1": [], "Subspan": []} for key in output_keys}

    for i in tqdm(range(total), desc="Evaluating"):
        answers = dataset[i]["answers"]
        for key in output_keys:
            raw_pred = preds_df.loc[i, key]
            pred = raw_pred.strip() if isinstance(raw_pred, str) else ""

            if not pred:
                em = f1 = subspan = 0.0
            else:
                em_scores = []
                f1_scores = []

                for ref in answers:
                    squad_result = squad_metric.compute(
                        predictions=[{"id": "x", "prediction_text": pred}],
                        references=[{"id": "x", "answers": {"text": [ref], "answer_start": [0]}}]
                    )
                    em_scores.append(squad_result["exact_match"])
                    f1_scores.append(squad_result["f1"])

                em = max(em_scores)
                f1 = max(f1_scores)
                subspan = compute_subspan_em(pred, answers)

            metrics[key]["EM"].append(em)
            metrics[key]["F1"].append(f1)
            metrics[key]["Subspan"].append(subspan)


    # 求每列平均值
    result = {}
    for key in output_keys:
        result[key] = {
            "EM": round(np.mean(metrics[key]["EM"]), 2),
            "F1": round(np.mean(metrics[key]["F1"]), 2),
            "Subspan EM": round(np.mean(metrics[key]["Subspan"]) * 100, 2),  # 乘 100
        }
    return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to input json dataset file (HF format)')
    parser.add_argument('--prediction', type=str, required=True, help='Path to csv file with model outputs')
    parser.add_argument('--num_dups', type=int, default=5, help='Number of output duplicates per example')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(">> Loading dataset...")
    ds = load_dataset("json", data_files=args.dataset, split="train")
    preds = pd.read_csv(args.prediction)

    output_keys = [f"output_{i}" for i in range(args.num_dups)]

    print(">> Running evaluation...")
    result = evaluate(preds, ds, output_keys)

    print("\n===== Evaluation Results =====")
    for key in output_keys:
        print(f"\nResult for {key}:")
        for metric, score in result[key].items():
            print(f"  {metric}: {score}%")
