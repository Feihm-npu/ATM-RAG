from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
from datasets import load_dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
import numpy as np
# from unsloth.utils.ropes import extend_qwen_rope
import wandb
# PatchDPOTrainer()
import importlib
import transformers

# importlib.reload(transformers.models.llama.modeling_llama)
# importlib.reload(transformers.models.qwen2.modeling_qwen2)
# PatchDPOTrainer()
from trl import DPOTrainer

truncation_mode = 'keep_end'
label_pad_token_id = -100
max_prompt_length = 3072
max_length = 4096

def build_tokenized_answer(prompt, answer, tokenizer):
    """
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """

    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )

def tokenize_row(feature, tokenizer):
    """Tokenize a single row from a DPO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + chosen or prompt + rejected responses is/are too long. First
    we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
    the sum of the length of the prompt and the chosen/rejected response, with
    label_pad_token_id  for the prompt tokens.
    """
    batch = {}
    prompt = feature["prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]


    if not isinstance(prompt, str):
        raise ValueError(f"prompt should be an str but got {type(prompt)}")
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

    if not isinstance(chosen, str):
        raise ValueError(f"chosen should be an str but got {type(chosen)}")
    chosen_tokens = build_tokenized_answer(prompt, chosen, tokenizer)

    if not isinstance(rejected, str):
        raise ValueError(f"rejected should be an str but got {type(rejected)}")
    rejected_tokens = build_tokenized_answer(prompt, rejected, tokenizer)

    # Last prompt token might get merged by tokenizer and
    # it should not be included for generation if that happens
    prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

    chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
    rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
    prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

    for k, v in prompt_tokens.items():
        prompt_tokens[k] = v[:prompt_len_input_ids]

    # Make sure prompts only have one different token at most an
    # and length only differs by 1 at most
    num_diff_tokens = sum(
        [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
    )
    num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
    if num_diff_tokens > 1 or num_diff_len > 1:
        raise ValueError(
            "Chosen and rejected prompt_input_ids might only differ on the "
            "last token due to tokenizer merge ops."
        )

    # add BOS token to head of prompt
    if tokenizer.bos_token_id is not None:
        prompt_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
        chosen_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
        rejected_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

    # add EOS token to end of answer
    chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
    chosen_tokens["attention_mask"].append(1)

    rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
    rejected_tokens["attention_mask"].append(1)

    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    # if combined sequence is too long, truncate the prompt
    for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
            if truncation_mode == "keep_start":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: max_prompt_length]
            elif truncation_mode == "keep_end":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][-max_prompt_length :]
            else:
                raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    # if that's still too long, truncate the response
    for answer_tokens in [chosen_tokens, rejected_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
            for k in ["input_ids", "attention_mask"]:
                answer_tokens[k] = answer_tokens[k][: max_length - max_prompt_length]

    # Create labels
    chosen_sequence_tokens = {
        k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    rejected_sequence_tokens = {
        k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
        label_pad_token_id
    ] * len(chosen_tokens["prompt_input_ids"])
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
        label_pad_token_id
    ] * len(rejected_tokens["prompt_input_ids"])

    for k, toks in {
        "chosen_": chosen_sequence_tokens,
        "rejected_": rejected_sequence_tokens,
        "": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}{type_key}"] = tokens

    # print(f"From tokenize_row tokenize_row output: {batch}")  # 调试信息

    return batch


# Step 1: Load model and tokenizer
max_seq_length = 2048
load_in_4bit = False
# model_name = "Qwen/Qwen2.5-7B"
model_name = "unsloth/Qwen2.5-7B"

dtype = None  # Auto detect based on GPU

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
# if hasattr(model.model, "rotary_emb") and hasattr(model.model.rotary_emb, "current_rope_size"):
#     model.model.rotary_emb.current_rope_size = max_seq_length

# tokenizer.pad_token = tokenizer.eos_token
# Step 2: Apply LoRA (or other PEFT method)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# Step 3: Load and tokenize dataset
# NQ_dpo.jsonl is the dataset from NQ and built based on the generator's perference, to train the adversary model.
raw_dataset = load_dataset("json", data_files="/home/feihm/llm-fei/Data/ATM/test_data_with_fabs/NQ/NQ_dpo.jsonl", split="train")

# processed_dataset = raw_dataset.map(
#     tokenize_row,
#     fn_kwargs={"tokenizer": tokenizer},
#     num_proc=8,
#     remove_columns=raw_dataset.column_names,
# )

# Step 4: Set training arguments
training_args = TrainingArguments(
    output_dir="./adv_outputs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    weight_decay=1e-4,
    num_train_epochs=1,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=1,
    report_to="tensorboard",
    gradient_accumulation_steps=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    save_strategy="no",
    dataloader_num_workers=1,
)

# Step 5: Initialize trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Will be loaded internally
    # model_init_kwargs={"torch_dtype": torch.bfloat16},
    # ref_model_init_kwargs={"torch_dtype": torch.bfloat16},
    args=training_args,
    beta=0.2,
    train_dataset=raw_dataset,
    eval_dataset=raw_dataset,
    tokenizer=tokenizer,
    max_length=2048,
    processing_class=tokenize_row,
    max_prompt_length=1536,
    dataset_num_proc=8,
)

# Step 6: Train
trainer.train()
# trainer.save_model("./outputs/model_1epoch_dpo")
# 保存 adapter（LoRA 参数）
model.save_pretrained("./s2_experiments/model_1epoch_dpo")

# 保存 tokenizer（否则下一次 tokenizer 也加载不了）
tokenizer.save_pretrained("./s2_experiments/model_1epoch_dpo")
