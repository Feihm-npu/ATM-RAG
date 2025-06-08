import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from accelerate import Accelerator
import json
import os
from tqdm import tqdm
import gc
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_LENGTH = 2048


class AnswerLossTrainer(Trainer):
    """自定义Trainer用于计算answer-level的loss"""
    
    def __init__(self, save_answer_losses=True, detail_output_path=None, **kwargs):
        super().__init__(**kwargs)
        self.save_answer_losses = save_answer_losses
        self.detail_output_path = detail_output_path
        self.detail_file = None
        self.sample_counter = 0
        
        # 如果需要保存详细信息，打开文件
        if self.save_answer_losses and self.detail_output_path:
            self.detail_file = open(f"{self.detail_output_path}_answer_details.jsonl", 'w', encoding='utf-8')
    
    def __del__(self):
        """确保文件被正确关闭"""
        if self.detail_file:
            self.detail_file.close()
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            # 使用混合精度加速
            with torch.amp.autocast('cuda', enabled=self.args.fp16 or self.args.bf16):
                outputs = model(**inputs)
                logits = outputs.logits
            
            labels = inputs['labels']
            
            # 计算token级别的loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 使用更高效的loss计算
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            ).view(shift_labels.shape)
            
            # 计算每个回答的平均loss
            valid_tokens = (shift_labels != -100).float()
            valid_counts = valid_tokens.sum(dim=1)
            
            # 避免除零错误
            valid_counts = torch.clamp(valid_counts, min=1.0)
            answer_loss_per_sample = (token_losses * valid_tokens).sum(dim=1) / valid_counts
            
            # 保存详细信息（如果需要）- 在GPU上进行
            if self.save_answer_losses and self.detail_file:
                self._save_batch_details(token_losses, shift_labels, answer_loss_per_sample, inputs)
            
            # 及时释放GPU内存
            del outputs, logits, shift_logits, token_losses
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 保持在GPU上，让Trainer自己处理gather
            return (None, answer_loss_per_sample, None)
    
    def _save_batch_details(self, token_losses, shift_labels, answer_loss_per_sample, inputs):
        """保存批次的详细信息到文件"""
        batch_size = token_losses.shape[0]
        
        for i in range(batch_size):
            valid_mask = shift_labels[i] != -100
            if valid_mask.any():
                valid_losses = token_losses[i][valid_mask].cpu().numpy()
                valid_tokens_ids = shift_labels[i][valid_mask].cpu().numpy()
                
                # 解码文本（避免OOM）
                try:
                    answer_text = self.tokenizer.decode(valid_tokens_ids, skip_special_tokens=True)
                except Exception as e:
                    answer_text = f"[Decode Error: {str(e)}]"
                
                detail_info = {
                    'sample_idx': self.sample_counter,
                    'answer_loss': float(answer_loss_per_sample[i].item()),
                    'answer_length': int(len(valid_losses)),
                    'answer_text': answer_text[:500],  # 限制长度
                    'mean_token_loss': float(valid_losses.mean()),
                    'max_token_loss': float(valid_losses.max()),
                    'min_token_loss': float(valid_losses.min()),
                }
                
                # 添加元数据（如果存在）
                if 'original_idx' in inputs:
                    detail_info['original_idx'] = int(inputs['original_idx'][i].item())
                if 'dup_idx' in inputs:
                    detail_info['dup_idx'] = int(inputs['dup_idx'][i].item())
                
                self.detail_file.write(json.dumps(detail_info, ensure_ascii=False) + '\n')
                self.sample_counter += 1
        
        # 定期刷新文件缓冲区
        if self.sample_counter % 100 == 0:
            self.detail_file.flush()


@dataclass
class DataCollatorWithLabelPad:
    """自定义数据整理器，支持labels的padding"""
    tokenizer: Any
    pad_to_multiple_of: int = 8
    include_metadata: bool = True
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 分离元数据和特征
        metadata_keys = ['original_idx', 'dup_idx']
        metadata = {}
        
        if self.include_metadata:
            for key in metadata_keys:
                if key in features[0]:
                    metadata[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
                    for f in features:
                        f.pop(key, None)
        
        # 提取labels
        labels = [f.pop("labels") for f in features]
        
        # padding input_ids和attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # padding labels，使用-100作为padding token
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label_list in labels:
            pad_length = max_len - len(label_list)
            if pad_length > 0:
                padded_label = label_list + [-100] * pad_length
            else:
                padded_label = label_list[:max_len]  # 截断（虽然不应该发生）
            padded_labels.append(padded_label)
        
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        # 添加元数据
        batch.update(metadata)
        
        return batch


def template_from_file(example, args):
    """将原始数据转换为模型输入格式"""
    item = {}
    item['answer'] = example['answers'][0] if example['answers'] else ""
    item['question'] = example['question']
    
    # 为每个context创建一个输入
    example['input'] = []
    example['target'] = []
    
    ctxs = example.get('ctxs', [])
    if not ctxs:
        return example
    
    # 限制context数量
    for ctx in ctxs[:args.num_dups]:
        title = ctx.get('title', 'Unknown')
        text = ctx.get('text', '')
        
        paragraph = f"[document] # Title: {title} ## text: {text} [/document]"
        prompt = (
            "You are a helpful assistant. Below is a question and some retrieved documents (some may be irrelevant).\n"
            "Use them to write a high-quality, concise, and accurate answer.\n\n"
            "[Knowledge]\n"
            f"{paragraph}\n"
            f"Question: {item['question']}\n\nAnswer:\n"
        )
        
        example['input'].append(prompt)
        example['target'].append(item['answer'])
    
    return example


def format_tokenize_row(example, tokenizer):
    """Tokenize并格式化单个样本"""
    assert tokenizer.padding_side == 'left'
    
    # 处理输入和目标
    inputs = example['input']
    targets = example['target']
    
    # 如果是空的，返回空列表
    if not inputs:
        return {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
    
    # 分别tokenize input和target
    input_encs = tokenizer(
        inputs,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH-256,
        padding=False
    )
    
    target_encs = tokenizer(
        targets,
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        padding=False
    )
    
    # 准备输出
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    # 处理每个context
    for i in range(len(inputs)):
        # 组合input + target + eos
        input_ids = input_encs['input_ids'][i] + target_encs['input_ids'][i] + [tokenizer.eos_token_id]
        
        # 如果超长，截断input部分
        if len(input_ids) > MAX_LENGTH:
            max_input_length = MAX_LENGTH - len(target_encs['input_ids'][i]) - 1
            input_ids = input_encs['input_ids'][i][:max_input_length] + target_encs['input_ids'][i] + [tokenizer.eos_token_id]
        
        # 创建attention mask
        attention_mask = [1] * len(input_ids)
        
        # 创建labels：input部分为-100，target部分保持原值
        input_length = len(input_ids) - len(target_encs['input_ids'][i]) - 1
        labels = [-100] * input_length + target_encs['input_ids'][i] + [tokenizer.eos_token_id]
        
        # 确保长度一致
        assert len(input_ids) == len(labels) == len(attention_mask)
        
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)
    
    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
        'labels': all_labels
    }


def flatten_and_add_metadata(examples, indices):
    """展平数据并添加元数据"""
    flattened = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'original_idx': [],
        'dup_idx': []
    }
    
    for i, idx in enumerate(indices):
        # 每个样本可能有多个contexts
        for dup_idx, (input_ids, attention_mask, labels) in enumerate(
            zip(examples['input_ids'][i], examples['attention_mask'][i], examples['labels'][i])
        ):
            flattened['input_ids'].append(input_ids)
            flattened['attention_mask'].append(attention_mask)
            flattened['labels'].append(labels)
            flattened['original_idx'].append(idx)
            flattened['dup_idx'].append(dup_idx)
    
    return flattened


def parse_args():
    p = argparse.ArgumentParser(description="Calculate per-context answer loss for RAG evaluation")
    p.add_argument("--model_name_or_path", required=True, help="Path to the model")
    p.add_argument("--input_file", required=True, help="Input JSON file with questions and contexts")
    p.add_argument("--output", required=True, help="Output CSV file for losses")
    p.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size for evaluation")
    p.add_argument("--num_procs", type=int, default=8, help="Number of processes for data preprocessing")
    p.add_argument("--num_dups", type=int, default=5, help="Number of contexts per question")
    p.add_argument("--save_answer_details", action="store_true", help="Save detailed answer-level losses")
    p.add_argument("--max_samples", type=int, default=-1, help="Maximum samples to process (-1 for all)")
    p.add_argument("--test100", action="store_true", help="Use only 100 samples for testing")
    p.add_argument("--fp16", action="store_true", help="Use fp16 precision")
    p.add_argument("--bf16", action="store_true", help="Use bf16 precision")
    p.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 设置环境变量以避免tokenizers警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 设置随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 初始化accelerator
    accelerator = Accelerator(
        mixed_precision='bf16' if args.bf16 else ('fp16' if args.fp16 else 'no'),
    )
    device = accelerator.device
    logger.info(f"Running on {accelerator.num_processes} processes")
    logger.info(f"Device: {device}")
    
    # 初始化tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True
    )
    
    # 设置tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 初始化模型
    logger.info(f"Loading model from {args.model_name_or_path}")
    
    # 动态设置dtype
    compute_dtype = torch.float32
    if args.fp16:
        compute_dtype = torch.float16
    elif args.bf16:
        compute_dtype = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )
    
    # 设置为评估模式
    model.eval()
    
    # 加载数据
    logger.info(f"Loading dataset from {args.input_file}")
    ds = load_dataset("json", data_files=args.input_file, split="train")
    
    # 应用样本限制
    if args.test100:
        ds = ds.select(range(min(100, len(ds))))
        logger.info("Test mode: Limited dataset to 100 samples")
    elif args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        logger.info(f"Limited dataset to {args.max_samples} samples")
    
    num_original_samples = len(ds)
    
    # 数据处理 - 使用accelerator.main_process_first()确保只处理一次
    with accelerator.main_process_first():
        # Step 1: 准备输入格式
        ds = ds.map(
            lambda x: template_from_file(x, args),
            num_proc=args.num_procs,
            desc="Preparing templates"
        )
        
        # 过滤掉没有contexts的样本
        ds = ds.filter(lambda x: len(x['input']) > 0)
        
        # Step 2: Tokenization
        ds = ds.map(
            lambda x: format_tokenize_row(x, tokenizer),
            num_proc=1,  # tokenizer不要并行
            remove_columns=ds.column_names,
            desc="Tokenizing"
        )
        
        # Step 3: 展平数据并添加元数据
        ds = ds.map(
            flatten_and_add_metadata,
            with_indices=True,
            batched=True,
            batch_size=1000,
            remove_columns=ds.column_names,
            desc="Flattening dataset"
        )
        
        logger.info(f"Dataset prepared. Total samples: {len(ds)}")
    
    # 设置训练参数
    output_dir = os.path.dirname(args.output) or "./eval_output"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        remove_unused_columns=False,
        report_to="none",
        prediction_loss_only=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_drop_last=False,
        label_names=["labels"],
        eval_accumulation_steps=None,
    )
    
    # 初始化trainer
    detail_output_path = args.output.replace('.csv', '') if args.save_answer_details else None
    
    trainer = AnswerLossTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithLabelPad(
            tokenizer, 
            pad_to_multiple_of=8,
            include_metadata=True
        ),
        tokenizer=tokenizer,
        eval_dataset=ds,
        save_answer_losses=args.save_answer_details,
        detail_output_path=detail_output_path,
    )
    
    # 执行预测
    logger.info("Starting loss calculation...")
    results = trainer.predict(ds)
    
    # 等待所有进程完成
    accelerator.wait_for_everyone()
    
    # 处理和保存结果（只在主进程中执行）
    if trainer.is_world_process_zero():
        try:
            # 获取预测结果
            predictions = results.predictions
            logger.info(f"Total predictions: {len(predictions)}")
            
            # 重塑预测结果
            pred_matrix = predictions.reshape((num_original_samples, args.num_dups))
            
            # 创建DataFrame
            column_names = [f'ctx_{idx}_loss' for idx in range(args.num_dups)]
            df = pd.DataFrame(pred_matrix, columns=column_names)
            
            # 保存结果
            df.to_csv(args.output, index=False, float_format='%.6f')
            logger.info(f"Saved results to {args.output}")
            
            # 打印统计信息
            logger.info("\n=== Results Summary ===")
            logger.info(f"Processed samples: {num_original_samples}")
            logger.info(f"Contexts per sample: {args.num_dups}")
            logger.info(f"Overall mean loss: {pred_matrix.mean():.4f}")
            logger.info(f"Overall std loss: {pred_matrix.std():.4f}")
            logger.info("\nPer-context statistics:")
            for i in range(args.num_dups):
                logger.info(f"  Context {i}: mean={pred_matrix[:, i].mean():.4f}, std={pred_matrix[:, i].std():.4f}")
            
            # 如果保存了详细信息，打印路径
            if args.save_answer_details and detail_output_path:
                logger.info(f"\nDetailed answer losses saved to: {detail_output_path}_answer_details.jsonl")
                
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            raise
        finally:
            # 确保关闭详细信息文件
            if hasattr(trainer, 'detail_file') and trainer.detail_file:
                trainer.detail_file.close()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()