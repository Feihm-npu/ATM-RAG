# mito_trainer.py

from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase
from trl import DPOTrainer
from trl.models import create_reference_model
from trl.trainer.utils import DPODataCollatorWithPadding, disable_dropout_in_model, pad_to_length


class MITOTrainer(DPOTrainer):
    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset,
        tokenizer,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid",
        data_collator=None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        reference_free: bool = False,
        **kwargs,
    ):
        self.beta = beta
        self.loss_type = loss_type
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.kldiv_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

        self.is_encoder_decoder = hasattr(model.config, "is_encoder_decoder") and model.config.is_encoder_decoder
        self.ref_model = None if reference_free else create_reference_model(model)

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=self.tokenizer.pad_token_id,
                label_pad_token_id=self.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
            args.remove_unused_columns = False

        disable_dropout_in_model(model)
        if self.ref_model:
            disable_dropout_in_model(self.ref_model)

        super().__init__(
            model=model,
            args=args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=None,
        )

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        model_outputs = model(
            concatenated_batch["concatenated_input_ids"],
            labels=concatenated_batch["concatenated_labels"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )

        all_logits = model_outputs.logits
    
        vocab_size = all_logits.shape[-1]
        batch_size = all_logits.shape[0]

        # Shift so that tokens < n predict n
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = concatenated_batch["concatenated_labels"][..., 1:].contiguous()
        # Flatten the tokens

        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        all_losses = self.ce_loss(shift_logits, shift_labels)
        all_losses = all_losses.view(batch_size, -1)
        all_losses = torch.sum(all_losses, dim=-1) / torch.sum((all_losses >= 1e-9).to(torch.long), dim=-1)

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_loss = all_losses[:len_chosen]
        rejected_loss = all_losses[len_chosen:]

        model_loss = model_outputs.loss
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_loss, rejected_loss, model_loss)


    def get_batch_logps(self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id
        print("→ non-masked token count:", loss_mask.sum())
        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}

        def get_pad_value(field_name):
            if "labels" in field_name:
                return label_pad_token_id
            elif "attention_mask" in field_name:
                return 0
            elif "input_ids" in field_name:
                return padding_value
            else:
                raise ValueError(f"Unexpected key {field_name}")

        # get all keys from "chosen_" prefix
        keys_to_pad = set(k.replace("chosen_", "") for k in batch if k.startswith("chosen_"))

        for key in keys_to_pad:
            chosen_k = f"chosen_{key}"
            rejected_k = f"rejected_{key}"
            concat_k = f"concatenated_{key}"

            max_len = max(batch[chosen_k].shape[1], batch[rejected_k].shape[1])
            pad_value = get_pad_value(key)

            chosen_padded = pad_to_length(batch[chosen_k], max_len, pad_value=pad_value)
            rejected_padded = pad_to_length(batch[rejected_k], max_len, pad_value=pad_value)

            concatenated = torch.cat([chosen_padded, rejected_padded], dim=0).to(device)
            concatenated_batch[concat_k] = concatenated

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1).to(device)

        # 修复 labels 长度不一致
        if "concatenated_labels" in concatenated_batch:
            input_len = concatenated_batch["concatenated_input_ids"].shape[1]
            label_len = concatenated_batch["concatenated_labels"].shape[1]
            if label_len < input_len:
                pad_len = input_len - label_len
                pad = torch.full(
                    (concatenated_batch["concatenated_labels"].shape[0], pad_len),
                    label_pad_token_id,
                    dtype=concatenated_batch["concatenated_labels"].dtype,
                    device=concatenated_batch["concatenated_labels"].device,
                )
                concatenated_batch["concatenated_labels"] = torch.cat(
                    [concatenated_batch["concatenated_labels"], pad], dim=1
                )
            elif label_len > input_len:
                concatenated_batch["concatenated_labels"] = concatenated_batch["concatenated_labels"][:, :input_len]

        return concatenated_batch





    def mito_loss(self, pred_logits, target_logits, pred_labels, target_labels):
        # Align shape
        max_len = max(pred_logits.size(1), target_logits.size(1))
        
        def pad_to(tensor, length, pad_value):
            pad_len = length - tensor.size(1)
            if pad_len > 0:
                pad_shape = list(tensor.shape[:-2]) + [pad_len, tensor.size(-1)]
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, padding], dim=1)
            return tensor[:, :length, :]

        def pad_labels(labels, length):
            pad_len = length - labels.size(1)
            if pad_len > 0:
                padding = torch.full((labels.size(0), pad_len), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([labels, padding], dim=1)
            return labels[:, :length]

        pred_logits = pad_to(pred_logits, max_len, torch.finfo(pred_logits.dtype).min)
        target_logits = pad_to(target_logits, max_len, torch.finfo(target_logits.dtype).min)
        pred_labels = pad_labels(pred_labels, max_len)
        target_labels = pad_labels(target_labels, max_len)

        # Create mask
        mask = (pred_labels != -100) & (target_labels != -100)

        pred_prob = F.log_softmax(pred_logits, dim=-1)
        target_prob = F.softmax(target_logits, dim=-1)

        # KL divergence per token
        kl = F.kl_div(pred_prob, target_prob, reduction="none").sum(dim=-1)  # shape: (B, L)
        kl = kl * mask

        return kl.sum() / mask.sum().clamp(min=1)


    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_loss,
            policy_rejected_loss,
            policy_model_loss,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            reference_chosen_logits,
                            reference_rejected_logits,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        reference_chosen_logits,
                        reference_rejected_logits,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        # adversarial

        chosen_sft_losses = policy_chosen_loss

        rejected_sft_losses = policy_rejected_loss

        pol_kl_losses = self.mito_loss(
            # policy_chosen_logits,
            policy_chosen_logits,
            policy_rejected_logits,
            batch['chosen_labels'],
            batch['rejected_labels']
            # reference_rejected_logits,
        )

        ref_kl_losses = self.mito_loss(
            # policy_chosen_logits,
            reference_chosen_logits,
            reference_rejected_logits,
            batch['chosen_labels'],
            batch['rejected_labels']
            # reference_rejected_logits,
        )
        
        # losses = kl_losses + policy_chosen_loss + policy_rejected_loss
        losses =  policy_chosen_loss + self.beta * (pol_kl_losses - ref_kl_losses)

        # rejected_diff = policy_rejected_logps - reference_chosen_logps
        # chosen_logps_diff_loss =  - F.logsigmoid(chosen_diff)
        # losses = chosen_logps_diff_loss / chosen_logps_diff_loss.detach() + kl_losses / kl_losses.detach()
        # losses = kl_losses +  chosen_logps_diff_loss
        # reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        # metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        # metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        # metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        # metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()

        metrics[f"{prefix}logits/pol_chosen"] = policy_chosen_logits.mean().cpu()
        metrics[f"{prefix}logits/pol_rejected"] = policy_rejected_logits.mean().cpu()
        metrics[f"{prefix}logits/ref_chosen"] = reference_chosen_logits.mean().cpu()
        metrics[f"{prefix}logits/ref_rejected"] = reference_rejected_logits.mean().cpu()

        metrics[f"{prefix}logits/pol_diff"] = policy_chosen_logits.detach().mean().cpu() - policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/ref_diff"] = reference_chosen_logits.detach().mean().cpu() - reference_rejected_logits.detach().mean().cpu()

        metrics[f"{prefix}logps/pol_diff"] = policy_chosen_logps.detach().mean().cpu() - policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_diff"] = reference_chosen_logps.detach().mean().cpu() - reference_rejected_logps.detach().mean().cpu()


        metrics[f"{prefix}logps/pol_rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/pol_chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_rejected"] = reference_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_chosen"] = reference_chosen_logps.detach().mean().cpu()

        metrics[f"{prefix}logits/pol_rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/pol_chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/ref_rejected"] = reference_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/ref_chosen"] = reference_chosen_logits.detach().mean().cpu()
        
        metrics[f"{prefix}sft_loss/pol_chosen"] = policy_chosen_loss.detach().mean().cpu()
        metrics[f"{prefix}sft_loss/pol_rejected"] = policy_rejected_loss.detach().mean().cpu()

        return losses.mean(), metrics


def mito_tokenize_row(feature, tokenizer, max_length=2048):
    prompt, adv_prompt, answer = feature["prompt"], feature["adv_prompt"], feature["answer"]
    eos = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    label_pad_token_id = -100

    prompt_enc = tokenizer([prompt, adv_prompt], add_special_tokens=False)
    answer_enc = tokenizer(answer, add_special_tokens=False)
    answer_ids = answer_enc["input_ids"] + [eos]
    answer_mask = answer_enc["attention_mask"] + [1]

    def build(prompt_ids, prompt_mask):
        input_ids = prompt_ids + answer_ids
        attention_mask = prompt_mask + answer_mask
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            input_ids = input_ids[overflow:]
            attention_mask = attention_mask[overflow:]
            prompt_len = max(len(prompt_ids) - overflow, 0)
        else:
            prompt_len = len(prompt_ids)
        labels = input_ids[:]
        labels[:prompt_len] = [label_pad_token_id] * prompt_len
        non_masked = sum(1 for x in labels if x != label_pad_token_id)
        print(f"[DEBUG] prompt_len={len(prompt_ids)}, answer_len={len(answer_ids)}, total={len(input_ids)}, labels!=-100: {non_masked}")

        assert any(l != label_pad_token_id for l in labels), "All labels are masked!"
        return input_ids, attention_mask, labels

    c_ids, c_mask, c_labels = build(prompt_enc["input_ids"][0], prompt_enc["attention_mask"][0])
    r_ids, r_mask, r_labels = build(prompt_enc["input_ids"][1], prompt_enc["attention_mask"][1])

    return {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
    }


@dataclass
class MITODataCollatorWithPadding:
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded = {}
        for k in features[0]:
            if k.endswith(("input_ids", "attention_mask", "labels")):
                seqs = [torch.tensor(ex[k]) for ex in features]
                padding_value = self.pad_token_id if "input_ids" in k else (
                    self.label_pad_token_id if "labels" in k else 0)
                padded[k] = pad_sequence(seqs, batch_first=True, padding_value=padding_value)
            else:
                padded[k] = [ex[k] for ex in features]
        return padded
