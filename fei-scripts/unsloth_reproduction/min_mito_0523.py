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


class minimal_MITOTrainer(DPOTrainer):
    def __init__(self, 
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, 
                 args: Optional[DPOConfig] = None, 
                 **kwargs 
                ):
        super().__init__(model=model, ref_model=ref_model, args=args, **kwargs)
        
        self.mito_alpha = getattr(self.args, "mito_alpha", self.args.beta) 
        if self.mito_alpha is None:
            print(f'Setting self.mito_alpha to 0.1')
            self.mito_alpha = 0.1
        if not hasattr(self.args, "mito_alpha") and self.is_world_process_zero():
            warnings.warn(
                f"`mito_alpha` not found in DPOConfig, using `beta` ({self.args.beta}) from DPOConfig instead. "
                f"Set `mito_alpha` in DPOConfig for explicit control."
            )
        
        self.sft_on_d_prime = getattr(self.args, "sft_on_d_prime", False)
        if self.is_world_process_zero():
            if hasattr(self.args, "sft_on_d_prime"):
                sft_path_info = "D' (rejected/adv_prompt path)" if self.sft_on_d_prime else "D (chosen/prompt path)"
                print(f"MITO SFT loss will be calculated on {sft_path_info} (sft_on_d_prime={self.sft_on_d_prime})")
            else:
                warnings.warn(
                    f"`sft_on_d_prime` not found in DPOConfig (self.args). Defaulting to True (SFT on D'). "
                    f"To control this, add `sft_on_d_prime: bool` to your DPOConfig or training arguments."
                )


    @staticmethod
    def tokenize_row(
        features: Dict[str, str], 
        processing_class: PreTrainedTokenizerBase, 
        max_prompt_length: int, 
        max_completion_length: int, 
        add_special_tokens: bool = False 
    ) -> Dict[str, List[int]]:
        tokenizer = processing_class

        answer_token_ids = tokenizer(features["prompt"], add_special_tokens=False, truncation=False)["input_ids"]
        if max_prompt_length is not None:
            if len(answer_token_ids) > max_prompt_length -1: 
                 answer_token_ids = answer_token_ids[:max_prompt_length -1]
        answer_tokens = answer_token_ids + [tokenizer.eos_token_id]

        context_d_tokens = tokenizer(features["chosen"], add_special_tokens=False, truncation=False)["input_ids"]
        if max_completion_length is not None: 
            context_d_tokens = context_d_tokens[:max_completion_length]

        context_d_prime_tokens = tokenizer(features["rejected"], add_special_tokens=False, truncation=False)["input_ids"]
        if max_completion_length is not None: 
            context_d_prime_tokens = context_d_prime_tokens[:max_completion_length]
            
        return {
            "prompt_input_ids": answer_tokens, 
            "prompt_attention_mask": [1] * len(answer_tokens),
            "chosen_input_ids": context_d_tokens, 
            "chosen_attention_mask": [1] * len(context_d_tokens),
            "rejected_input_ids": context_d_prime_tokens, 
            "rejected_attention_mask": [1] * len(context_d_prime_tokens),
        }


    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, torch.LongTensor], 
        padding_value: int,
    ) -> Dict[str, torch.LongTensor]:
        output = {}
        contexts_d = batch["chosen_input_ids"]
        contexts_d_prime = batch["rejected_input_ids"]
        contexts_d_mask = batch["chosen_attention_mask"]
        contexts_d_prime_mask = batch["rejected_attention_mask"]
        answer_ids = batch["prompt_input_ids"] 
        answer_mask = batch["prompt_attention_mask"]

        max_context_len = max(contexts_d.shape[1], contexts_d_prime.shape[1])
        output["prompt_input_ids"] = torch.cat(
            (
                pad_to_length_left(contexts_d, max_context_len, pad_value=padding_value, dim=1),
                pad_to_length_left(contexts_d_prime, max_context_len, pad_value=padding_value, dim=1),
            ),
            dim=0,
        )
        output["prompt_attention_mask"] = torch.cat(
            (
                pad_to_length_left(contexts_d_mask, max_context_len, pad_value=0, dim=1),
                pad_to_length_left(contexts_d_prime_mask, max_context_len, pad_value=0, dim=1),
            ),
            dim=0,
        )

        max_answer_len = answer_ids.shape[1]
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length_right(answer_ids, max_answer_len, pad_value=padding_value, dim=1),
                pad_to_length_right(answer_ids.clone(), max_answer_len, pad_value=padding_value, dim=1),
            ),
            dim=0,
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length_right(answer_mask, max_answer_len, pad_value=0, dim=1),
                pad_to_length_right(answer_mask.clone(), max_answer_len, pad_value=0, dim=1),
            ),
            dim=0,
        )
        return output

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, torch.LongTensor]):
        concatenated_batch_for_model = self.concatenated_inputs(batch, padding_value=self.padding_value)
        model_kwargs = {}
        # if self.args.output_router_logits: # User can uncomment if using MoE models
        #     model_kwargs["output_router_logits"] = True
        
        effective_prompt_input_ids = concatenated_batch_for_model["prompt_input_ids"]
        effective_prompt_attention_mask = concatenated_batch_for_model["prompt_attention_mask"]
        effective_completion_input_ids = concatenated_batch_for_model["completion_input_ids"]
        effective_completion_attention_mask = concatenated_batch_for_model["completion_attention_mask"]

        if self.is_encoder_decoder: 
            labels_for_model = effective_completion_input_ids.clone() 
            if self.label_pad_token_id != -100: 
                 labels_for_model[effective_completion_attention_mask == 0] = self.label_pad_token_id
            
            model_outputs = model(
                input_ids=effective_prompt_input_ids,
                attention_mask=effective_prompt_attention_mask,
                labels=labels_for_model,
                **model_kwargs,
            )
            raw_logits = model_outputs.logits
            final_loss_mask = effective_completion_attention_mask.bool()
            final_labels_for_logps = labels_for_model
        else: 
            input_ids = torch.cat((effective_prompt_input_ids, effective_completion_input_ids), dim=1)
            attention_mask = torch.cat((effective_prompt_attention_mask, effective_completion_attention_mask), dim=1)
            
            loss_mask_for_answer_part_of_sequence = torch.cat(
                (torch.zeros_like(effective_prompt_attention_mask), effective_completion_attention_mask),
                dim=1,
            )

            attention_mask, input_ids, loss_mask_for_answer_part_of_sequence = flush_left(
                attention_mask, input_ids, loss_mask_for_answer_part_of_sequence
            )

            if self.args.max_length is not None: 
                if self.args.truncation_mode == "keep_end":
                    input_ids = input_ids[:, -self.args.max_length :]
                    attention_mask = attention_mask[:, -self.args.max_length :]
                    loss_mask_for_answer_part_of_sequence = loss_mask_for_answer_part_of_sequence[:, -self.args.max_length :]
                elif self.args.truncation_mode == "keep_start":
                    input_ids = input_ids[:, : self.args.max_length]
                    attention_mask = attention_mask[:, : self.args.max_length]
                    loss_mask_for_answer_part_of_sequence = loss_mask_for_answer_part_of_sequence[:, : self.args.max_length]
                else: 
                    raise ValueError(f"Unknown truncation mode: {self.args.truncation_mode}")

            if self.args.padding_free: 
                raise NotImplementedError("Padding-free for overridden concatenated_forward not fully implemented here.")
            else:
                model_kwargs["attention_mask"] = attention_mask
            
            model_outputs = model(input_ids, **model_kwargs)
            raw_logits = model_outputs.logits

            final_labels_for_logps = torch.roll(input_ids, shifts=-1, dims=1)
            final_loss_mask = torch.roll(loss_mask_for_answer_part_of_sequence, shifts=-1, dims=1).bool()
            
            if raw_logits.shape[1] > final_labels_for_logps.shape[1]:
                 raw_logits = raw_logits[:, raw_logits.shape[1] - final_labels_for_logps.shape[1] :, :]
            elif raw_logits.shape[1] < final_labels_for_logps.shape[1]: 
                 final_labels_for_logps = final_labels_for_logps[:, final_labels_for_logps.shape[1] - raw_logits.shape[1] :, :]
                 final_loss_mask = final_loss_mask[:, final_loss_mask.shape[1] - raw_logits.shape[1] :, :]

        masked_labels_for_logps_calc = final_labels_for_logps.clone()
        masked_labels_for_logps_calc[~final_loss_mask] = 0 

        per_token_logps = selective_log_softmax(raw_logits, masked_labels_for_logps_calc)
        per_token_logps[~final_loss_mask] = 0 
        per_token_logps_for_sum = torch.roll(per_token_logps, shifts=1, dims=1)
        all_logps = (per_token_logps_for_sum * final_loss_mask).sum(-1)

        output_dict = {}
        output_dict["logits"] = raw_logits 
        output_dict["labels"] = final_labels_for_logps 
        output_dict["loss_mask"] = final_loss_mask 

        original_batch_size = raw_logits.shape[0] // 2
        output_dict["chosen_logps"] = all_logps[:original_batch_size]
        output_dict["rejected_logps"] = all_logps[original_batch_size:]
        
        loss_mask_chosen_path = final_loss_mask[:original_batch_size]
        loss_mask_rejected_path = final_loss_mask[original_batch_size:]
        logits_chosen_path = raw_logits[:original_batch_size]
        logits_rejected_path = raw_logits[original_batch_size:]

        if loss_mask_chosen_path.any():
            output_dict["mean_chosen_logits"] = logits_chosen_path[loss_mask_chosen_path].mean()
        else:
            output_dict["mean_chosen_logits"] = torch.tensor(0.0, device=raw_logits.device, dtype=raw_logits.dtype)

        if loss_mask_rejected_path.any():
            output_dict["mean_rejected_logits"] = logits_rejected_path[loss_mask_rejected_path].mean()
        else:
            output_dict["mean_rejected_logits"] = torch.tensor(0.0, device=raw_logits.device, dtype=raw_logits.dtype)

        # if self.args.output_router_logits and hasattr(model_outputs, "aux_loss"): # User can uncomment
        #     output_dict["aux_loss"] = model_outputs.aux_loss
            
        return output_dict

    def _calculate_kl_loss(
        self,
        pred_logits: torch.Tensor,    
        target_logits: torch.Tensor,  
        loss_mask: torch.Tensor       
    ) -> torch.Tensor:
        log_probs_pred = F.log_softmax(pred_logits, dim=-1) 
        probs_target = F.softmax(target_logits, dim=-1)    
        
        kl_div_per_token = F.kl_div(
            input=log_probs_pred, 
            target=probs_target, 
            reduction='none', 
            log_target=False  
        ).sum(dim=-1) 
        
        masked_kl_div = kl_div_per_token * loss_mask.bool() 

        num_answer_tokens = loss_mask.sum()
        if num_answer_tokens > 0:
            kl_loss = masked_kl_div.sum() / num_answer_tokens 
        else:
            kl_loss = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype)
        return kl_loss

    def get_batch_loss_metrics(
        self,
        model: nn.Module, 
        batch: Dict[str, torch.LongTensor], 
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        metrics = {}
        
        policy_output = self.concatenated_forward(model, batch) 

        num_original_examples = batch["prompt_input_ids"].shape[0]

        policy_all_logits = policy_output["logits"] 
        policy_all_labels = policy_output["labels"] 
        policy_all_loss_mask = policy_output["loss_mask"]

        policy_logits_d_plus_a = policy_all_logits[:num_original_examples]
        policy_logits_d_prime_plus_a = policy_all_logits[num_original_examples:]
        
        # Determine which labels to use for SFT based on self.sft_on_d_prime
        policy_labels_for_sft: torch.Tensor
        if self.sft_on_d_prime:
            policy_labels_for_sft = policy_all_labels[num_original_examples:] 
        else:
            policy_labels_for_sft = policy_all_labels[:num_original_examples]
            
        policy_loss_mask_a_in_d_plus_a = policy_all_loss_mask[:num_original_examples] # Mask for 'a' in D+a path (for KL_policy)

        ref_output = None
        kl_ref = torch.tensor(0.0, device=policy_all_logits.device, dtype=policy_all_logits.dtype)

        with torch.no_grad(): 
            if self.ref_model is None: 
                if self.is_peft_model:
                    with self.null_ref_context(): 
                        ref_output = self.concatenated_forward(self.model, batch)
                else:
                    if self.is_world_process_zero():
                        warnings.warn(
                            "ref_model is None and model is not PEFT. KL_ref will be zero. "
                            "Ensure DPOConfig and model setup correctly provide/create a reference model if KL difference is desired."
                        )
            else: 
                ref_output = self.concatenated_forward(self.ref_model, batch)

        if ref_output is not None:
            ref_all_logits = ref_output["logits"]
            ref_all_loss_mask = ref_output["loss_mask"] 

            ref_logits_d_plus_a = ref_all_logits[:num_original_examples]
            ref_logits_d_prime_plus_a = ref_all_logits[num_original_examples:]
            # Mask for 'a' in D+a path for reference model (for KL_ref)
            ref_loss_mask_a_in_d_plus_a = ref_all_loss_mask[:num_original_examples] 

            kl_ref = self._calculate_kl_loss(
                pred_logits=ref_logits_d_plus_a,
                target_logits=ref_logits_d_prime_plus_a,
                loss_mask=ref_loss_mask_a_in_d_plus_a 
            )
        
        # --- SFT Loss Calculation (conditional) ---
        sft_target_logits_for_loss: torch.Tensor
        sft_loss_metric_name: str

        if self.sft_on_d_prime:
            sft_target_logits_for_loss = policy_logits_d_prime_plus_a
            sft_loss_metric_name = "sft_on_D_prime" # D' is the rejected/adv_prompt path
        else:
            sft_target_logits_for_loss = policy_logits_d_plus_a
            sft_loss_metric_name = "sft_on_D" # D is the chosen/prompt path
        
        sft_logits_flat = sft_target_logits_for_loss.reshape(-1, sft_target_logits_for_loss.size(-1))
        # Use the correctly selected policy_labels_for_sft
        sft_labels_flat = policy_labels_for_sft.reshape(-1) 
        
        sft_loss = F.cross_entropy(
            sft_logits_flat,
            sft_labels_flat,
            ignore_index=self.args.label_pad_token_id, 
            reduction="mean" 
        )
        
        kl_policy = self._calculate_kl_loss(
            pred_logits=policy_logits_d_plus_a,
            target_logits=policy_logits_d_prime_plus_a,
            loss_mask=policy_loss_mask_a_in_d_plus_a 
        )

        total_loss = sft_loss + self.mito_alpha * (kl_policy - kl_ref)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}loss/{sft_loss_metric_name}"] = sft_loss.item() 
        metrics[f"{prefix}loss/kl_policy"] = kl_policy.item()
        if ref_output is not None: 
            metrics[f"{prefix}loss/kl_reference"] = kl_ref.item()
            metrics[f"{prefix}loss/kl_difference"] = (kl_policy - kl_ref).item()
        metrics[f"{prefix}loss/mito_total"] = total_loss.item() 

        metrics[f"{prefix}logps/answer_given_D_policy"] = policy_output["chosen_logps"].mean().item()
        metrics[f"{prefix}logps/answer_given_D_prime_policy"] = policy_output["rejected_logps"].mean().item()
        metrics[f"{prefix}mean_logits/answer_given_D_policy"] = policy_output["mean_chosen_logits"].mean().item()
        metrics[f"{prefix}mean_logits/answer_given_D_prime_policy"] = policy_output["mean_rejected_logits"].mean().item()
        
        if ref_output is not None:
            metrics[f"{prefix}logps/answer_given_D_ref"] = ref_output["chosen_logps"].mean().item()
            metrics[f"{prefix}logps/answer_given_D_prime_ref"] = ref_output["rejected_logps"].mean().item()
            metrics[f"{prefix}mean_logits/answer_given_D_ref"] = ref_output["mean_chosen_logits"].mean().item()
            metrics[f"{prefix}mean_logits/answer_given_D_prime_ref"] = ref_output["mean_rejected_logits"].mean().item()

        # if self.args.output_router_logits and "aux_loss" in policy_output: # User can uncomment
        #     aux_loss = policy_output["aux_loss"] 
        #     total_loss += self.args.router_aux_loss_coef * aux_loss 
        #     metrics[f"{prefix}loss/aux"] = aux_loss.item()
        #     metrics[f"{prefix}loss/mito_total_with_aux"] = total_loss.item()

        return total_loss, metrics
