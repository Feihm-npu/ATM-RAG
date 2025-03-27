import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    # trl_sanitze_kwargs_for_tagging,
)
from dataclasses import dataclass


class MITOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid",
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,    
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name
        self.reference_free = reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOTrainer's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')


        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.dataset_num_proc = dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        # with PartialState().local_main_process_first():
        #     # tokenize the dataset
        #     train_dataset = train_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)
        #     if eval_dataset is not None:
        #         eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)

        Trainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)


    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        concatenated_batch = self.concatenated_inputs(
            batch, padding_value=self.padding_value
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

        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = concatenated_batch["concatenated_labels"][..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)

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
        del model_outputs, concatenated_batch, batch
        torch.cuda.empty_cache()

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_loss, rejected_loss, model_loss)

    def get_batch_logps(
    self,
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    is_encoder_decoder: bool = False,
    label_pad_token_id: int = -100,
) -> torch.FloatTensor:
        """
        计算 batch 中每个样本的 log-likelihood 总和或平均值。

        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)
            average_log_prob: 是否对每个 sample 的 logprob 取平均，否则是 sum
            is_encoder_decoder: 用于确定是否处理 encoder-decoder 架构
            label_pad_token_id: 用于 mask 的 pad token

        Returns:
            log_probs: (batch_size,) 每个 sample 的对数概率
        """
        # shift for causal lm
        if not is_encoder_decoder:
            logits = logits[:, :-1]
            labels = labels[:, 1:]

        # flatten for log_softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        vocab_size = logits.size(-1)

        # 获取 labels 的概率
        labels = labels.to(logits.device)
        label_mask = labels != label_pad_token_id

        # 将 labels 编码成 one-hot，得到每个位置的 log_p(label)
        labels_one_hot = torch.nn.functional.one_hot(labels * label_mask, num_classes=vocab_size).bool()
        log_p = log_probs[labels_one_hot].view(logits.size(0), -1)

        # 对每个 sample 求和或平均
        if average_log_prob:
            log_p = log_p.sum(dim=1) / label_mask.sum(dim=1)
        else:
            log_p = log_p.sum(dim=1)

        return log_p


    def concatenated_inputs(
    self,
    batch: Dict[str, Union[List, torch.LongTensor]],
    padding_value: int = 0,
) -> Dict[str, torch.Tensor]:
        """
        Concatenates chosen and rejected inputs into a single batch for MITO training.
        """
        # 1. Extract tensors
        chosen_input_ids = batch["chosen_input_ids"]
        rejected_input_ids = batch["rejected_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        rejected_attention_mask = batch["rejected_attention_mask"]
        chosen_labels = batch["chosen_labels"]
        rejected_labels = batch["rejected_labels"]

        # 2. Concatenate along batch dim
        concatenated_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        concatenated_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        concatenated_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

        return {
            "concatenated_input_ids": concatenated_input_ids,
            "concatenated_attention_mask": concatenated_attention_mask,
            "concatenated_labels": concatenated_labels,
        }


    def get_batch_loss_metrics(self, model, batch, train_eval: Literal["train", "eval"] = "train"):
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

        with torch.no_grad():
            if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
                ref_chosen_logps = batch["reference_chosen_logps"]
                ref_rejected_logps = batch["reference_rejected_logps"]
                ref_chosen_logits = ref_rejected_logits = None
            elif self.ref_model is None:
                with self.null_ref_context():
                    (
                        ref_chosen_logps,
                        ref_rejected_logps,
                        ref_chosen_logits,
                        ref_rejected_logits,
                        _, _, _
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    ref_chosen_logps,
                    ref_rejected_logps,
                    ref_chosen_logits,
                    ref_rejected_logits,
                    _, _, _
                ) = self.concatenated_forward(self.ref_model, batch)

        pol_kl_loss = self.mito_loss(policy_chosen_logits, policy_rejected_logits, batch['chosen_labels'], batch['rejected_labels'])

        if ref_chosen_logits is not None and ref_rejected_logits is not None:
            ref_kl_loss = self.mito_loss(ref_chosen_logits, ref_rejected_logits, batch['chosen_labels'], batch['rejected_labels'])
        else:
            ref_kl_loss = torch.tensor(0.0, device=policy_chosen_loss.device)

        losses = policy_chosen_loss + self.beta * (pol_kl_loss - ref_kl_loss.detach())

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}sft_loss/pol_chosen"] = policy_chosen_loss.mean().detach().cpu()
        metrics[f"{prefix}sft_loss/pol_rejected"] = policy_rejected_loss.mean().detach().cpu()

        return losses.mean(), metrics


    def mito_loss(self, pred_logits, target_logits, pred_labels, target_labels):
        pred_logits = pred_logits.masked_fill((pred_labels == -100).unsqueeze(-1), torch.finfo(pred_logits.dtype).min)
        target_logits = target_logits.masked_fill((target_labels == -100).unsqueeze(-1), torch.finfo(target_logits.dtype).min)

        tar_prob = F.softmax(target_logits.flatten(0, 1), dim=1)
        pred_prob = F.log_softmax(pred_logits.flatten(0, 1), dim=1)

        kl_loss = self.kldiv_loss(pred_prob, tar_prob)
        return kl_loss
        

def mito_tokenize_row(feature, tokenizer, max_total_length=4096) -> Dict:
    """
    Tokenize and truncate a single data point for MITO training.

    This version enforces truncation after prompt+answer concatenation,
    ensuring the total length is within `max_total_length`.
    """
    label_pad_token_id = -100
    prompt = feature["prompt"]
    adv_prompt = feature["adv_prompt"]
    answer = feature["answer"]

    if not isinstance(prompt, str) or not isinstance(adv_prompt, str) or not isinstance(answer, str):
        raise ValueError("All of prompt, adv_prompt, and answer must be strings.")

    assert tokenizer.padding_side == "left"

    # Encode prompts (no special tokens)
    prompt_encs = tokenizer([prompt, adv_prompt], add_special_tokens=False)
    # Encode answer (also no special tokens)
    answer_enc = tokenizer(answer, add_special_tokens=False)
    answer_input_ids = answer_enc["input_ids"] + [tokenizer.eos_token_id]
    answer_attention_mask = answer_enc["attention_mask"] + [1]

    def truncate_and_build(prompt_ids, prompt_mask):
        input_ids = prompt_ids + answer_input_ids
        attention_mask = prompt_mask + answer_attention_mask

        # Truncate from the left if too long
        if len(input_ids) > max_total_length:
            overflow = len(input_ids) - max_total_length
            input_ids = input_ids[overflow:]
            attention_mask = attention_mask[overflow:]
            prompt_len = max(len(prompt_ids) - overflow, 0)
        else:
            prompt_len = len(prompt_ids)

        labels = input_ids[:]
        labels[:prompt_len] = [label_pad_token_id] * prompt_len
        return input_ids, attention_mask, labels

    chosen_input_ids, chosen_attention_mask, chosen_labels = truncate_and_build(
        prompt_encs["input_ids"][0], prompt_encs["attention_mask"][0]
    )
    rejected_input_ids, rejected_attention_mask, rejected_labels = truncate_and_build(
        prompt_encs["input_ids"][1], prompt_encs["attention_mask"][1]
    )

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_labels": rejected_labels,
    }



    
@dataclass
class MITODataCollatorWithPadding:
    r"""
    MITO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch

