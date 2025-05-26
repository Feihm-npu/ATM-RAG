import inspect
# import unsloth # Not directly used by this class, but by the training script
# from unsloth import FastLanguageModel # Not directly used by this class
import os
import random
import textwrap
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union, Dict, List, Tuple

import pandas as pd
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence # Not directly used
import transformers
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm # tqdm might be used by base
from datasets import Dataset, IterableDataset
from packaging import version
from torch.utils.data import DataLoader
from transformers import (
    # AutoModelForCausalLM, # Not directly used by this class
    # BaseImageProcessor, # For type hints if vision models were supported
    DataCollator, # For type hint in __init__
    # FeatureExtractionMixin, # For type hints if vision models were supported
    PreTrainedModel, # For type hints
    PreTrainedTokenizerBase, # For type hints
    # ProcessorMixin, # For type hints if vision models were supported
    # Trainer, # Base class is DPOTrainer
    is_comet_available, # For logging if used by base
    is_wandb_available, # For logging if used by base
)
from transformers.data.data_collator import DataCollatorMixin # Used by DPOTrainer
# from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES # If vision
from transformers.trainer_callback import TrainerCallback # If callbacks are used
from transformers.trainer_utils import EvalLoopOutput # For prediction_step type hint
from transformers.utils import is_peft_available, is_torch_xpu_available # For PEFT/XPU checks

from trl import DPOTrainer 
from trl.trainer import DPOConfig 
# from trl.trainer.dpo_trainer import DataCollatorForPreference # Not used, custom collator expected
# from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt # Not used due to pre-tokenization
# from trl.trainer.utils import ( # Some might be used by inherited methods if not fully overridden
#     RunningMoments,
#     cap_exp,
#     disable_dropout_in_model,
#     empty_cache,
#     flush_left, # Not used by this new logic path
#     generate_model_card,
#     get_comet_experiment_url,
#     log_table_to_comet_experiment,
#     pad,
#     pad_to_length, # The specific version without padding_side is the user's concern
#     peft_module_casting_to_bf16,
#     selective_log_softmax, # Not used by this new logic path
# )


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

class min_MITOTrainer(DPOTrainer):
    r"""
    min_MITOTrainer for implementing the MITO loss for the Generator.
    This trainer expects pre-tokenized datasets with specific columns like:
    'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels',
    'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'.
    The 'chosen_labels' and 'rejected_labels' should correspond to the 'answer' part,
    with context parts masked using label_pad_token_id.

    MITO Loss = SFT_loss(answer | D', q) + alpha * KL_loss( P(answer|D,q) || P(answer|D',q) )
    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[DPOConfig] = None, 
        data_collator: Optional[DataCollator] = None, # Should be DataCollatorForMITO
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None, # Tokenizer
        # ref_model is intentionally omitted as it's not used by MITO loss
        **kwargs, 
    ):
        # MITO does not use a reference model in the DPO sense.
        # Explicitly pass ref_model=None to the parent.
        if "ref_model" in kwargs and kwargs["ref_model"] is not None:
            warnings.warn(
                "min_MITOTrainer does not use a `ref_model` for its loss computation. "
                "The provided `ref_model` will be ignored by setting it to None for the parent."
            )
        kwargs["ref_model"] = None # Ensure ref_model is None for DPOTrainer init

        if args is not None and hasattr(args, 'precompute_ref_log_probs'):
            if args.precompute_ref_log_probs:
                warnings.warn("`precompute_ref_log_probs` is set to True in DPOConfig but is not used by min_MITOTrainer. Setting to False.")
            args.precompute_ref_log_probs = False # Not used by MITO

        super().__init__(
            model=model,
            # ref_model=None, # Already handled by kwargs modification
            args=args,
            data_collator=data_collator, 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs
        )
        
        # Ensure label_pad_token_id is set, defaulting to -100
        if self.label_pad_token_id is None: 
            self.label_pad_token_id = -100 # Common default for HF CrossEntropyLoss
            if self.is_world_process_zero():
                warnings.warn("`label_pad_token_id` was not set in DPOConfig, defaulting to -100 for min_MITOTrainer.")

        # SFT loss function (CrossEntropyLoss)
        # Reduction is 'mean' by default, which is standard.
        self.sft_loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id) 

        # Alpha for weighting the KL term in MITO loss.
        # Fetches 'mito_alpha' from DPOConfig. If not present, defaults to 0.1.
        # The training script (dpo_test.py) should ensure 'mito_alpha' is part of DPOConfig.
        # Alternatively, could use self.args.beta if that's the intended parameter.
        self.mito_alpha = getattr(self.args, "mito_alpha", 0.1) 
        if not hasattr(self.args, "mito_alpha") and self.is_world_process_zero(): 
             warnings.warn(
                 f"`mito_alpha` not found in DPOConfig (self.args). Using default value: {self.mito_alpha}. "
                 f"Consider adding `mito_alpha` to your DPOConfig or training arguments."
            )

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Optional[PreTrainedTokenizerBase], # processing_class is tokenizer
        args: DPOConfig, 
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        """
        Prepares the dataset. For min_MITOTrainer, this method expects that
        the dataset has already been tokenized by an external process (e.g., a script
        that uses a `mito_tokenize_row`-like function) and contains all necessary
        columns like 'chosen_input_ids', 'chosen_labels', etc.
        """
        if dataset is None:
            return None

        # Define the columns expected from the pre-tokenization step
        required_cols = [
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels"
        ]
        
        is_already_tokenized = False
        if isinstance(dataset, Dataset):
            # Check if all required columns are present
            is_already_tokenized = all(col in dataset.column_names for col in required_cols)
        elif isinstance(dataset, IterableDataset):
            # For IterableDataset, we can't easily check column names beforehand.
            # We assume it's correctly formatted or rely on errors during collation.
            # Alternatively, one could take a sample if possible, but that's complex.
            if self.is_world_process_zero():
                warnings.warn(
                    f"Dataset '{dataset_name}' is an IterableDataset. Assuming it is correctly pre-tokenized "
                    f"with columns: {', '.join(required_cols)}. If not, errors may occur later."
                )
            # For simplicity, let's assume if it's iterable, it's correctly formatted.
            # A more robust check might be needed depending on the IterableDataset source.
            # For now, we can proceed as if it might be tokenized.
            # A more direct approach for IterableDataset is to skip this check or
            # have a flag in DPOConfig indicating pre-tokenization.
            # Let's assume if it's iterable, it's correctly formatted by the user.
            pass # No easy check, proceed with caution or user must ensure format.


        if is_already_tokenized or isinstance(dataset, IterableDataset): # Assume iterable is prepped
            if self.is_world_process_zero():
                print(f"Dataset '{dataset_name}' is assumed to be pre-tokenized for MITO. Skipping internal tokenization.")
            return dataset
        else:
            # If not pre-tokenized and not iterable, it's an error for this trainer version
            # as it does not provide its own `tokenize_row`.
            raise ValueError(
                f"Dataset '{dataset_name}' is not pre-tokenized with the required columns for min_MITOTrainer "
                f"(missing one or more of: {', '.join(required_cols)}) and this trainer version "
                f"does not implement `tokenize_row`. Please pre-tokenize your dataset using a "
                f"`mito_tokenize_row`-like function that produces these columns."
            )

    # get_train_dataloader and get_eval_dataloader are overridden to ensure
    # they use the provided self.data_collator (expected to be DataCollatorForMITO)
    # and standard DataLoader logic, bypassing potential Unsloth DPO-specifics if any.
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator 

        if not isinstance(data_collator, DataCollatorForMITO):
            warnings.warn(f"Expected self.data_collator to be DataCollatorForMITO, but got {type(data_collator)}. Ensure it's correctly set.")

        if isinstance(train_dataset, IterableDataset):
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size, # From DPOConfig
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size, # This is effective batch size on device
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        current_eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator 

        if not isinstance(data_collator, DataCollatorForMITO):
            warnings.warn(f"Expected self.data_collator to be DataCollatorForMITO, but got {type(data_collator)}. Ensure it's correctly set.")


        if isinstance(current_eval_dataset, IterableDataset):
            return DataLoader(
                current_eval_dataset,
                batch_size=self.args.eval_batch_size, # From DPOConfig
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(current_eval_dataset)

        return DataLoader(
            current_eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size, 
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _calculate_kl_loss(
        self, 
        pred_logits: torch.Tensor, # Logits from D+a path
        target_logits: torch.Tensor, # Logits from D'+a path
        pred_labels: torch.Tensor, # Labels for 'a' in D+a (masked for context D)
        target_labels: torch.Tensor # Labels for 'a' in D'+a (masked for context D')
    ) -> torch.Tensor:
        """
        Calculates KL(P_pred || P_target) where P_pred is P(a|D) and P_target is P(a|D').
        KL divergence is computed only on tokens where both pred_labels and target_labels
        are valid (not label_pad_token_id).
        """
        # Logits and labels are expected to be of shape (batch_size, seq_len, vocab_size/1)
        # seq_len here is the length of the D+a or D'+a sequence.
        
        # Ensure sequences are padded to the same length for KL calculation
        # This is important if D and D' have different lengths, leading to
        # D+a and D'+a having different lengths before collation.
        # DataCollatorForMITO should have already padded them to max_length_in_batch.
        # So, pred_logits and target_logits should have the same seq_len here.
        if pred_logits.size(1) != target_logits.size(1):
            # This case should ideally not happen if DataCollatorForMITO works correctly.
            # If it does, padding here is a fallback.
            max_len = max(pred_logits.size(1), target_logits.size(1))
            
            def _pad_tensor_to(tensor, length, pad_value, is_labels=False):
                pad_len = length - tensor.size(1)
                if pad_len > 0:
                    if is_labels:
                        pad_shape = (tensor.size(0), pad_len)
                        padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                    else: # logits
                        pad_shape = (tensor.size(0), pad_len, tensor.size(-1))
                        padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                    tensor = torch.cat([tensor, padding], dim=1) # Right padding
                return tensor[:, :length] if not is_labels else tensor[:, :length]

            pred_logits = _pad_tensor_to(pred_logits, max_len, 0.0)
            target_logits = _pad_tensor_to(target_logits, max_len, 0.0)
            pred_labels = _pad_tensor_to(pred_labels, max_len, self.label_pad_token_id, is_labels=True)
            target_labels = _pad_tensor_to(target_labels, max_len, self.label_pad_token_id, is_labels=True)

        # Mask for valid answer tokens where KL should be computed
        # (i.e., tokens that are part of 'a' and not padding, in both paths)
        mask = (pred_labels != self.label_pad_token_id) & \
               (target_labels != self.label_pad_token_id)
        
        if not mask.any(): 
            return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype)

        log_probs_pred = F.log_softmax(pred_logits, dim=-1) # log P(a|D)
        probs_target = F.softmax(target_logits, dim=-1)    # P(a|D')

        # KL divergence D_KL(P_pred || P_target)
        kl_div_per_token = F.kl_div(
            input=log_probs_pred, 
            target=probs_target, 
            reduction='none', # Keep per-element KL
            log_target=False  # target is not log-probabilities
        ).sum(dim=-1) # Sum over the vocabulary dimension -> shape (batch_size, seq_len)
        
        masked_kl_div = kl_div_per_token * mask # Apply mask

        num_active_tokens = mask.sum()
        if num_active_tokens > 0:
            kl_loss = masked_kl_div.sum() / num_active_tokens # Average over active tokens
        else:
            kl_loss = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype)
            
        return kl_loss

    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        batch: Dict[str, torch.LongTensor], # Output of DataCollatorForMITO
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the MITO loss and metrics for a batch.
        The batch contains pre-tokenized and collated 'chosen_*' and 'rejected_*' fields.
        'chosen_*' fields correspond to the (D, a) path.
        'rejected_*' fields correspond to the (D', a) path.
        Labels ('chosen_labels', 'rejected_labels') are for the 'a' part, with context masked.
        """
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        
        # Forward pass for the "chosen" path (D + a)
        # Input: D+a, Labels: for 'a' part of D+a
        outputs_chosen_path = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            # labels=batch["chosen_labels"] # Not passing labels here to get raw logits
                                        # Loss will be calculated manually.
        )
        logits_chosen_path = outputs_chosen_path.logits # Logits for P(token | D, a_<token>)

        # Forward pass for the "rejected" path (D' + a)
        # Input: D'+a, Labels: for 'a' part of D'+a
        outputs_rejected_path = model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            # labels=batch["rejected_labels"] # Not passing labels here for raw logits
        )
        logits_rejected_path = outputs_rejected_path.logits # Logits for P(token | D', a_<token>)

        # 1. SFT Loss: L_SFT(a | D')
        # Calculated on the "rejected" path (negative context D').
        # Reshape for CrossEntropyLoss: (N*S, C) and (N*S)
        sft_logits_flat = logits_rejected_path.reshape(-1, logits_rejected_path.size(-1))
        sft_labels_flat = batch["rejected_labels"].reshape(-1)
        
        sft_loss = self.sft_loss_fct(sft_logits_flat, sft_labels_flat) # ignore_index handled by self.sft_loss_fct
        metrics[f"{prefix}loss/sft"] = sft_loss.item()

        # 2. KL Loss Term: KL(P_G(a|D) || P_G(a|D'))
        kl_loss = self._calculate_kl_loss(
            pred_logits=logits_chosen_path,      # Logits from D+a path
            target_logits=logits_rejected_path,  # Logits from D'+a path
            pred_labels=batch["chosen_labels"],    # Labels for 'a' in D+a (for masking)
            target_labels=batch["rejected_labels"] # Labels for 'a' in D'+a (for masking)
        )
        metrics[f"{prefix}loss/kl"] = kl_loss.item()
        
        # Total MITO loss
        total_loss = sft_loss + self.mito_alpha * kl_loss
        metrics[f"{prefix}loss/total_mito"] = total_loss.item() # Log the primary loss
        
        # Optional: Log perplexity or other metrics if needed
        # For example, perplexity of the SFT loss part:
        # with torch.no_grad():
        #     metrics[f"{prefix}perplexity/sft"] = torch.exp(sft_loss).item()

        return total_loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]], # Batch from DataCollatorForMITO
        return_outputs=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Overrides Trainer.compute_loss to use custom MITO loss calculation.
        """
        # Note: The `num_items_in_batch` argument from the original signature is removed
        # as it's not typically used when providing a custom loss calculation like this.
        # If your Transformers version's Trainer.compute_loss expects it, add it back.

        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Store metrics for logging (DPOTrainer's mechanism)
        # self.store_metrics is available from DPOTrainer -> Trainer
        if self.is_world_process_zero(): # Only log metrics from the main process
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]], # Batch from DataCollatorForMITO
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Overrides Trainer.prediction_step for evaluation.
        """
        if ignore_keys is None:
            if hasattr(model, "config") and model.config is not None: # Added model.config is not None check
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        # Ensure model is in eval mode
        model.eval()

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # Store metrics for logging
        if self.is_world_process_zero():
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None
        
        # For EvalLoopOutput, loss, logits, labels are expected.
        # We return the main loss. For logits and labels, we can pass dummy tensors
        # as the primary evaluation is through the 'metrics' dict from store_metrics.
        # The actual model logits are too large and complex to return directly in a simple format here.
        # We can use some representative values from metrics if needed, or just placeholders.
        # Example: using sft_loss and kl_loss as representative "logits" for EvalPrediction.
        eval_sft_loss = metrics.get(f"eval_loss/sft", 0.0)
        eval_kl_loss = metrics.get(f"eval_loss/kl", 0.0)
        
        # Create a small tensor for the "logits" part of EvalLoopOutput if required by callbacks
        # This is just a placeholder.
        dummy_logits_output = torch.tensor([eval_sft_loss, eval_kl_loss], device=loss.device)
        # Labels also need to be a placeholder if not directly relevant for metric computation via EvalPrediction
        dummy_labels_output = torch.zeros(dummy_logits_output.shape[0], dtype=torch.long, device=loss.device)

        return (loss.detach(), dummy_logits_output, dummy_labels_output)


@dataclass
class DataCollatorForMITO:
    """
    Data collator for MITO training. Assumes input features are already tokenized
    by an external `mito_tokenize_row`-like function, producing lists of token IDs and labels.
    This collator handles dynamic padding within each batch.
    """
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100 # Default for HF CrossEntropyLoss

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        # These are the keys expected from the pre-tokenized features
        keys_to_pad = [
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels"
        ]

        for key in keys_to_pad:
            sequences = [feature[key] for feature in features] # List of lists of ints

            if "labels" in key:
                current_padding_value = self.label_pad_token_id
            elif "attention_mask" in key:
                current_padding_value = 0     
            else:  # "input_ids"
                current_padding_value = self.tokenizer.pad_token_id
            
            if current_padding_value is None and "input_ids" in key:
                raise ValueError(
                    f"Padding value for {key} is None. Ensure tokenizer.pad_token_id is set."
                )

            # Determine padding side
            # For input_ids and attention_mask, usually left-padding is preferred for decoder-only models
            # to allow for more efficient generation/processing if sequences are very different in length.
            # However, for labels, right-padding is standard.
            # The `mito_tokenize_row` should have handled the internal structure (e.g., context + answer).
            # Here, we just pad the entire sequence.
            # Let's assume right padding for simplicity unless tokenizer.padding_side is explicitly left.
            padding_side = self.tokenizer.padding_side if hasattr(self.tokenizer, 'padding_side') else "right"
            if "labels" in key: # Labels are always right-padded for standard loss functions
                padding_side_for_key = "right"
            else:
                padding_side_for_key = padding_side


            max_length_in_batch = 0
            if sequences and sequences[0] is not None : #Check if sequences is not empty and first element is not None
                 max_length_in_batch = max(len(seq) for seq in sequences if seq is not None) # Ensure seq is not None
            else: # Handle empty sequences case
                batch[key] = torch.empty(0, dtype=torch.long) # Or handle as an error
                continue


            padded_sequences_list = []
            for seq in sequences:
                if seq is None: # Handle None sequences if they can occur
                    # This should ideally not happen if dataset is clean
                    seq = [] # Replace None with empty list for padding
                
                padding_length = max_length_in_batch - len(seq)
                if padding_side_for_key == "right":
                    padded_sequence = seq + [current_padding_value] * padding_length
                else:  # padding_side == "left"
                    padded_sequence = [current_padding_value] * padding_length + seq
                padded_sequences_list.append(padded_sequence)
            
            try:
                batch[key] = torch.tensor(padded_sequences_list, dtype=torch.long)
            except Exception as e:
                print(f"Error converting key {key} to tensor. Padded list: {padded_sequences_list}")
                raise e
            
        return batch
