import inspect
# import unsloth # Not used in the core logic for now
# from unsloth import FastLanguageModel # Not used
import os
import random
import textwrap
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext # nullcontext is used by DPOTrainer's null_ref_context
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union, Dict, List, Tuple

import pandas as pd
import torch
import torch.amp as amp # For autocast if needed, DPOTrainer handles this
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence # Not directly used, TRL utils used

import transformers
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm # tqdm might be useful
from datasets import Dataset, IterableDataset
from packaging import version
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    # Trainer, # DPOTrainer inherits from Trainer
    is_comet_available, # For logging
    is_wandb_available, # For logging
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_peft_available, is_torch_xpu_available

from trl import DPOTrainer, DPOConfig
# from trl.trainer.dpo_trainer import DataCollatorForPreference # Using this or a custom one
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
from trl.trainer.utils import ( # Import specific utilities that are still used
    flush_left, 
    selective_log_softmax,
    # pad_to_length # User has custom padding or version without padding_side
)
from trl.models import create_reference_model # Needed if not PEFT and no ref_model provided


def pad_to_length_left(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        # Truncate from the left if longer
        # Original TRL's pad_to_length keeps the end part of the sequence if truncating from left.
        # This implementation will take the last 'length' elements.
        return tensor.narrow(dim, tensor.size(dim) - length, length)
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                torch.full(pad_size, pad_value, dtype=tensor.dtype, device=tensor.device), 
                tensor,
            ],
            dim=dim,
        )

def pad_to_length_right(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        # Truncate from the right if longer
        return tensor.narrow(dim, 0, length)
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                torch.full(pad_size, pad_value, dtype=tensor.dtype, device=tensor.device), 
            ],
            dim=dim,
        )


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

@dataclass
class MITOConfig(DPOConfig):
    """
    Configuration class for minimal_MITOTrainer, extending DPOConfig.
    """
    sft_on_d_prime: bool = field(default=True, metadata={
        "help": "If True, SFT loss is calculated on the D' (rejected/adv_prompt) path. "
                "If False, SFT loss is calculated on the D (chosen/prompt) path."
    })
    mito_alpha: Optional[float] = field(default=None, metadata={
        "help": "Alpha weighting for the KL term in MITO loss. If None, defaults to DPOConfig's beta."
    })
    # You can add other MITO-specific parameters here if needed

# -----------------------------------------------------------------------------
# Trainer ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

class minimal_MITOTrainer(DPOTrainer):
    """Minimal, syntax‑correct MITO implementation re‑using TRL internals."""

    # ---------------------------------------------------------------------
    # Init -----------------------------------------------------------------
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.mito_alpha = getattr(self.args, "mito_alpha", None) or self.args.beta
        self.sft_on_d_prime = getattr(self.args, "sft_on_d_prime", False)
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)  # 修复：使用标准的ignore_index
        if self.is_world_process_zero():
            which = "D'" if self.sft_on_d_prime else "D"
            print(f"[MITO] mito_alpha={self.mito_alpha} | SFT on {which} path")

    # ---------------------------------------------------------------------
    # Tokenise one row ------------------------------------------------------
    # ---------------------------------------------------------------------
    @staticmethod
    def tokenize_row(
        features: Dict[str, str],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: Optional[int],
        max_completion_length: Optional[int],
        add_special_tokens: bool = False,
    ) -> Dict[str, List[int]]:
        tok = processing_class  # must be provided by TRL

        # Answer a ---------------------------------------------------------
        ans_ids = tok(features["prompt"], add_special_tokens=False)["input_ids"]
        ## NOT truncate the answer.
        # if max_prompt_length and len(ans_ids) > max_prompt_length - 1:
        #     ans_ids = ans_ids[: max_prompt_length - 1]
        ans_ids.append(tok.eos_token_id)

        # Contexts ---------------------------------------------------------
        def _ctx(text: str):
            ids = tok(text, add_special_tokens=False)["input_ids"]
            # leave *one* token slack so answer is never pushed out by flush_left/keep_end
            if max_prompt_length and len(ids) >= max_prompt_length:
                ids = ids[-(max_prompt_length - 1):]
            return ids

        ctx_D = _ctx(features["chosen"])      # original docs
        ctx_Dp = _ctx(features["rejected"])   # attacked docs

        return {
            "prompt_input_ids": ans_ids,
            "prompt_attention_mask": [1] * len(ans_ids),
            "chosen_input_ids": ctx_D,
            "chosen_attention_mask": [1] * len(ctx_D),
            "rejected_input_ids": ctx_Dp,
            "rejected_attention_mask": [1] * len(ctx_Dp),
        }

    # ---------------------------------------------------------------------
    # Concatenation helpers ------------------------------------------------
    # ---------------------------------------------------------------------
    @staticmethod
    def concatenated_inputs(batch: Dict[str, torch.Tensor], pad_val: int):
        out: Dict[str, torch.Tensor] = {}
        cd, cdp = batch["chosen_input_ids"], batch["rejected_input_ids"]
        md, mdp = batch["chosen_attention_mask"], batch["rejected_attention_mask"]
        # ans, ans_mask = batch["prompt_input_ids"], batch["prompt_attention_mask"]

        Lc = max(cd.shape[1], cdp.shape[1])
        out["prompt_input_ids"] = torch.cat([
            pad_to_length_left(cd, Lc, pad_val, 1),
            pad_to_length_left(cdp, Lc, pad_val, 1),
        ])
        out["prompt_attention_mask"] = torch.cat([
            pad_to_length_left(md, Lc, 0, 1),
            pad_to_length_left(mdp, Lc, 0, 1),
        ])

        out["completion_input_ids"] = torch.cat([
            batch["prompt_input_ids"], batch["prompt_input_ids"]
        ], dim=0)
        out["completion_attention_mask"] = torch.cat([
            batch["prompt_attention_mask"], batch["prompt_attention_mask"]
        ], dim=0)
        return out

    # ---------------------------------------------------------------------
    # Forward pass ---------------------------------------------------------
    # ---------------------------------------------------------------------
    def concatenated_forward(self, model: nn.Module, batch):
        x = self.concatenated_inputs(batch, self.padding_value)
        p_ids, p_mask = x["prompt_input_ids"], x["prompt_attention_mask"]
        a_ids, a_mask = x["completion_input_ids"], x["completion_attention_mask"]

        # 记录原始长度
        original_doc_len = p_ids.shape[1]
        original_ans_len = a_ids.shape[1]
        
        
        ids = torch.cat([p_ids, a_ids], 1)
        att = torch.cat([p_mask, a_mask], 1)
        mask = torch.cat([torch.zeros_like(p_mask), a_mask], 1)     
        
        # 记录截断前的信息
        truncated = False
        truncate_offset = 0
        
        if self.args.max_length and ids.shape[1] > self.args.max_length:
            truncated = True
            if self.args.truncation_mode == "keep_end":
                truncate_offset = ids.shape[1] - self.args.max_length
                ids, att, mask = (t[:, -self.args.max_length :] for t in (ids, att, mask))
            else:
                ids, att, mask = (t[:, : self.args.max_length] for t in (ids, att, mask))
        
        out = model(ids, attention_mask=att)
        logits = out.logits
        
        # 计算答案在截断后序列中的位置
        if truncated and self.args.truncation_mode == "keep_end":
            answer_start_in_truncated = original_doc_len - truncate_offset
        else:
            answer_start_in_truncated = original_doc_len

        lbl_ids = torch.roll(ids, -1, 1)
        mask = torch.roll(mask, shifts=-1, dims=1)
        mask = mask.bool()
        
     
        masked_lbl = lbl_ids.clone()
        masked_lbl[~mask] = -100
        
        valid_labels = (masked_lbl != -100).sum()

        
        logps = selective_log_softmax(logits, lbl_ids)
        logps[~mask] = 0
        logps = torch.roll(logps, shifts=1, dims=1)

        all_lp = logps.sum(-1)
        half = logits.shape[0] // 2
        
        assert logits.shape[0] % 2 == 0, f"Batch size must be even, got {logits.shape[0]}"
        assert half > 0, "Batch size too small"
        
        return {
            "chosen_logps": logps[:half],
            "rejected_logps": logps[half:],
            "chosen_logits": logits[:half],
            "rejected_logits": logits[half:],
            "chosen_labels": masked_lbl[:half],
            "rejected_labels": masked_lbl[half:],
        }

    def mito_loss(
            self,
            pred_logits,  # 修复：现在接收的是logits而不是logps
            target_logits,
            pred_labels,
            target_labels,
        ) -> torch.FloatTensor:

        # 添加验证：检查输入形状
        assert pred_logits.dim() == 3, f"Expected 3D logits, got {pred_logits.dim()}D"
        assert target_logits.dim() == 3, f"Expected 3D logits, got {target_logits.dim()}D"

        pred_logits = pred_logits.masked_fill(
            (pred_labels == -100).unsqueeze(-1), torch.finfo(pred_logits.dtype).min
        )

        target_logits = target_logits.masked_fill(
            (target_labels == -100).unsqueeze(-1), torch.finfo(target_logits.dtype).min
        )

        tar_prob = target_logits.view(-1, target_logits.shape[-1]).contiguous()
        pred_prob = pred_logits.view(-1, pred_logits.shape[-1]).contiguous()
        
        tar_prob = F.softmax(tar_prob, dim=1)
        pred_prob = F.log_softmax(pred_prob, dim=1)
        
        kl_loss = self.kldiv_loss(pred_prob, tar_prob)
        
        # 添加验证：确保损失在合理范围内
        if kl_loss > 100:
            warnings.warn(f"KL loss is unusually high: {kl_loss.item()}")
        
        return kl_loss

    # ---------------------------------------------------------------------
    # Loss -----------------------------------------------------------------
    # ---------------------------------------------------------------------
    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ):
        output = self.concatenated_forward(model, batch)
        
        chosen_logps = output["chosen_logps"]
        rejected_logps = output["rejected_logps"]
        chosen_logits = output["chosen_logits"]
        rejected_logits = output["rejected_logits"]
        chosen_labels = output["chosen_labels"]
        rejected_labels = output["rejected_labels"]

        if self.sft_on_d_prime:
            # SFT loss on D' (rejected/adv_prompt)
            loss_sft = self.ce_loss(
                rejected_logits.reshape(-1, rejected_logits.size(-1)),
                rejected_labels.reshape(-1),
            )
        else:
            loss_sft = self.ce_loss(
                chosen_logits.reshape(-1, chosen_logits.size(-1)),
                chosen_labels.reshape(-1),
            )

        ###############ref_model##############
        compte_ref_context_manager = amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch)

        ref_chosen_logits = ref_model_output["chosen_logits"]
        ref_rejected_logits = ref_model_output["rejected_logits"]
        ######################################

        kl_pol = self.mito_loss(
            pred_logits=rejected_logits,  # 现在传入logits
            target_logits=chosen_logits,   # 现在传入logits
            pred_labels=rejected_labels,
            target_labels=chosen_labels,
        )

        kl_ref = self.mito_loss(
            pred_logits=ref_rejected_logits,  # 现在传入logits
            target_logits=ref_chosen_logits,   # 现在传入logits
            pred_labels=rejected_labels,
            target_labels=chosen_labels,
        )
        # total = loss_sft
        total = loss_sft + self.mito_alpha * (kl_pol-kl_ref)

        tag = "sft_d_prime" if self.sft_on_d_prime else "sft_d"
        
        p = "eval_" if train_eval == "eval" else ""
        metrics: Dict[str, Any] = {
            f"{p}loss/{tag}": loss_sft.item(),
            f"{p}loss/kl_policy": kl_pol.item(),
            f"{p}loss/kl_reference": kl_ref.item(),
            f"{p}loss/mito_total": total.item(),
        }
        
        # 添加验证：确保所有损失值都是有限的
        for key, value in metrics.items():
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"Non-finite loss detected: {key} = {value}")
        
        return total, metrics