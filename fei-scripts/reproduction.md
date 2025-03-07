## Data download
```bash
sh ~/llm-fei/FiD/get-data.sh
```
## Data preparation
```bash
sh ./build_test.sh
```

## Initial tuning of Generator
```bash
python ../atm_train/generator_sft/generator_sft_data_prepare.py
```

## Training
```bash
sh ./train.sh
```

## Attacker DPO Optimization
```bash
sh attacker_dpo_opt.sh
```
1. The codes need modifications and dependency fixing. In 
```python
from transformers.models.llama.modeling_llama import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Tuple, Union
```
2. There still exists some problem here since I am using LLama configuration for a Qwen model.