# SLIP: Sensor Language-Informed Pretraining

**Learning Transferable Sensor Models via Language-Informed Pretraining**

Yuliang Chen, Arvind Pillai, Yu Yvonne Wu, Tess Z. Griffin, Lisa Marsch, Michael V. Heinz, Nicholas C. Jacobson, Andrew Campbell

Dartmouth College

[[Paper]](asset/manuscript.pdf) [[Model]](https://huggingface.co/LeoChen085/SLIP) [[Dataset]](https://huggingface.co/datasets/LeoChen085/SlipDataset) [[Demo]](demo.ipynb)

---

## Overview

SLIP is an open-source framework for learning language-aligned sensor representations that generalize across diverse sensor setups. It integrates contrastive alignment with sensor-conditioned captioning, enabling both discriminative understanding and generative reasoning over multivariate time series from heterogeneous sensors.

**Key features:**
- **FlexMLP**: A weight-sharing patch embedding that dynamically adapts to different temporal resolutions and variable-length inputs without retraining
- **Repurposed decoder-only LLM**: Splits a pretrained Gemma-3-270M into a unimodal text encoder (first 12 layers) and a multimodal decoder (last 6 layers with cross-attention), enabling efficient sensor-conditioned text generation
- **Contrastive + Captioning pretraining**: Joint CLIP-style contrastive loss and autoregressive captioning loss for both discriminative and generative capabilities
- **Cross-domain transfer**: Pretrained on 600K+ sensor-caption pairs (~1B time points) spanning health, environment, IoT, energy, and transportation

**Results:**
- 77.14% average linear-probing accuracy across 11 datasets (5.93% relative improvement over baselines)
- 64.83% average accuracy on sensor-based question answering
- 0.887 BERTScore on sensor captioning

## Architecture

<p align="center">
  <img src="asset/SLIP_overview.pdf" width="90%">
</p>

SLIP comprises four components:
1. **Sensor Encoder** (120M params): Transformer with FlexMLP patch embedding and 2D RoPE for cross-sensor and long-range temporal interactions
2. **Sensor Pooler**: Attention pooling with 65 learnable queries (1 CLS + 64 caption tokens) compressing variable-length sensor tokens to fixed-size representations
3. **Text Encoder**: First 12 layers of Gemma-3-270M (last 4 layers unfrozen)
4. **Multimodal Decoder**: Last 6 layers of Gemma-3-270M extended with cross-attention for sensor-conditioned generation

Total: ~220M parameters, 67M trainable.

## Installation

### 1. Create a virtual environment

```bash
conda create -n slip python=3.10 -y
conda activate slip
```

### 2. Install PyTorch

Install PyTorch with CUDA support matching your system. See [pytorch.org](https://pytorch.org/get-started/locally/) for options.

```bash
# Example for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Download pretrained checkpoint

```python
from huggingface_hub import hf_hub_download

# Download the pretrained base model checkpoint
hf_hub_download("LeoChen085/SLIP", "SLIP_gemma270.pth", local_dir="ckpt")

# (Optional) Download task-specific finetuned checkpoints
for name in ["har", "sleep", "ecg", "tsqa", "caption"]:
    hf_hub_download("LeoChen085/SLIP", f"{name}.safetensors", local_dir="ckpt")
```

### 5. Verify installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from transformers import AutoModel; print('transformers OK')"
python -c "from datasets import load_dataset; print('datasets OK')"
```

## Project Structure

```
SLIP/
├── model_factory/
│   ├── SLIP.py                 # Main SLIP model (contrastive + captioning)
│   ├── ts_transformer.py       # Sensor encoder with FlexMLP & RoPE
│   ├── multimodal_gemma.py     # Gemma-3 repurposed as encoder-decoder
│   └── chatts.py               # ChatTS baseline wrapper
├── config/
│   ├── train.yaml              # Pretraining config
│   ├── sensor_lp.yaml          # Linear probing config
│   ├── sensor_zs.yaml          # Zero-shot config
│   ├── sft.yaml                # Supervised finetuning config
│   ├── model/slip.yaml         # SLIP model hyperparameters
│   ├── sensor_encoder/rope_vit_base.yaml
│   └── dataset/
│       ├── pretrain.yaml       # Pretraining dataset config
│       ├── lp_dataset.yaml     # 11 linear probe task configs
│       ├── zs_dataset.yaml     # Zero-shot dataset config
│       └── opentslm_qa.yaml    # QA dataset config
├── evaluation/
│   ├── sensor_eval_main.py     # Linear probing evaluation
│   ├── zs_eval_main.py         # Zero-shot retrieval evaluation
│   ├── sft.py                  # SFT training & QA/captioning evaluation
│   ├── engine_eval.py          # Low-level train/eval loops
│   └── text_generation.py      # LLM prompting baseline evaluation
├── util/
│   ├── dataset.py              # Dataset classes & collators (see below)
│   ├── data_augmentation.py    # Sensor data augmentations
│   ├── normalization.py        # Normalization utilities
│   ├── metrics.py              # Evaluation metrics
│   ├── pos_embed.py            # Positional embedding utilities
│   ├── lr_sched.py / lr_decay.py  # Learning rate scheduling
│   ├── head.py                 # Classification heads
│   ├── misc.py                 # Distributed training & checkpointing
│   └── serialize.py            # Serialization utilities
├── data_generator/
│   ├── caption_generator.py    # Generate sensor captions from attributes
│   ├── data_to_attribute.py    # Extract time series attributes (trend, seasonality, noise)
│   ├── text_aug.py             # LLM-based text augmentation (Qwen2-7B)
│   └── config/datagen_config.yaml
├── script/
│   ├── train/pretrain.sh       # Pretraining launch script
│   └── eval/
│       ├── lp.sh               # Linear probing eval (11 tasks)
│       ├── zs.sh               # Zero-shot eval (11 tasks)
│       └── sft.sh              # SFT eval (QA + captioning)
├── ckpt/                       # Model checkpoints
│   ├── SLIP_gemma270.pth       # Pretrained SLIP base
│   ├── har.safetensors         # SFT checkpoint for HAR-CoT
│   ├── sleep.safetensors       # SFT checkpoint for Sleep-CoT
│   ├── ecg.safetensors         # SFT checkpoint for ECG-QA-CoT
│   ├── tsqa.safetensors        # SFT checkpoint for TSQA
│   └── caption.safetensors     # SFT checkpoint for M4 captioning
├── train_main.py               # Pretraining entry point
├── train_engine.py             # Training loop implementation
└── requirement.txt
```

## Data Preparation

### Pretraining Data

SLIP pretrains on 600K+ sensor-caption pairs. The dataset is publicly available on Hugging Face:

You can download or load the dataset from Hugging Face: [https://huggingface.co/datasets/LeoChen085/SlipDataset](https://huggingface.co/datasets/LeoChen085/SlipDataset)

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Load the dataset
ds = load_dataset("LeoChen085/SlipDataset")

# Download meta.csv from HuggingFace Hub
meta_path = hf_hub_download(
    repo_id="LeoChen085/SlipDataset",
    filename="meta.csv",
    repo_type="dataset",
)
```


### Evaluation Datasets

The 11 evaluation datasets span four domains:

| Domain | Datasets |
|--------|----------|
| Activity Recognition | WISDM, UCI-HAR |
| Clinical Diagnosis | Stroke (PPG_CVA), Diabetes (PPG_DM), Hypertension (PPG_HTN), Sleep Stage (sleepEDF), Heart Condition (ptbxl) |
| Stress Prediction | WESAD, StudentLife |
| Urban Sensing | AsphaltObstacles, Beijing AQI |

Evaluation datasets are automatically downloaded from [LeoChen085/SlipDataset](https://huggingface.co/datasets/LeoChen085/SlipDataset) if not found locally. To use local data instead, place files under `data/lp_dataset/<task_name>/` (linear probing) or `data/zs_dataset/<task_name>/` (zero-shot).

### SFT Datasets

SFT datasets (HAR-CoT, Sleep-CoT, ECG-QA, TSQA, M4 Captioning) are hosted at [LeoChen085/SlipSFTDataset](https://huggingface.co/datasets/LeoChen085/SlipSFTDataset) and loaded automatically at runtime.

## Quick Start

See [demo.ipynb](demo.ipynb) for a complete walkthrough. Below are representative examples for each usage mode.

### Load Pretraining Dataset

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from util.dataset import SLIPhfDataset, SLIPCollator
from torch.utils.data import DataLoader

ds = load_dataset("LeoChen085/SlipDataset")
meta_path = hf_hub_download(
    repo_id="LeoChen085/SlipDataset", filename="meta.csv", repo_type="dataset"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")

dataset = SLIPhfDataset(
    hf_dataset=ds["train"],
    meta_info=meta_path,
    patch_size=None,        # None = flexi patch
    text_aug=True,
    is_normalize=True,
    sampling=True,
)
loader = DataLoader(dataset, batch_size=32, collate_fn=SLIPCollator(tokenizer, max_len=128))
```

### Load Evaluation Datasets

```python
from util.dataset import ZeroshotDataset, SLIPCollator, EvalDataset, EvalCollator
from functools import partial

# Zero-shot — auto-downloads from HuggingFace if local data is missing
zs_dataset = ZeroshotDataset(
    data_dir="PPG_CVA",
    patch_size=16,
    return_patch=True,
    is_normalize=True,
    hf_repo="LeoChen085/SlipDataset",
)
zs_loader = DataLoader(zs_dataset, batch_size=32, shuffle=False,
                       collate_fn=SLIPCollator(tokenizer, max_len=128))

# Linear probing — auto-downloads from HuggingFace if local data is missing
train_set = EvalDataset("PPG_CVA", is_train=True,  patch_size=16, is_normalize=True,
                        hf_repo="LeoChen085/SlipDataset")
test_set  = EvalDataset("PPG_CVA", is_train=False, patch_size=16, is_normalize=True,
                        hf_repo="LeoChen085/SlipDataset")
train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                          collate_fn=partial(EvalCollator, return_patch=True))
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False,
                          collate_fn=partial(EvalCollator, return_patch=True))
```

### Load SFT Dataset

```python
from util.dataset import SftDataset, SFTCollator

train_set = SftDataset("har_cot", split="val", hf_repo="LeoChen085/SlipSFTDataset")
loader = DataLoader(train_set, batch_size=32, shuffle=True,
                    collate_fn=SFTCollator(tokenizer))
```

### Load Model

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("LeoChen085/SLIP", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model.eval()
```

### Toy Example with Synthetic Tensors

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Build synthetic sensor data (flexi-patch format)
batch_size, num_vars, num_patches, patch_size = 2, 3, 10, 16
sensor_ids, sensor_masks, sensor_times = [], [], []
for _ in range(batch_size):
    vars_x, vars_m, vars_t = [], [], []
    for _ in range(num_vars):
        vars_x.append(torch.randn(num_patches, patch_size, device=device))
        vars_m.append(torch.ones(num_patches, patch_size, device=device))
        vars_t.append(
            torch.linspace(0, 1, num_patches, device=device)
            .unsqueeze(-1).expand(num_patches, patch_size)
        )
    sensor_ids.append(vars_x)
    sensor_masks.append(vars_m)
    sensor_times.append(vars_t)

sensors = {
    "input_ids": sensor_ids,
    "attention_mask": sensor_masks,
    "time_index": sensor_times,
}

# Tokenize text queries
queries = ["Describe the pattern of this sensor data.", "What activity is this?"]
tok = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=64)
text = {k: v.to(device) for k, v in tok.items()}
```

### Get Contrastive Embeddings

```python
with torch.no_grad():
    text_emb, sensor_emb = model.get_embedding(text, sensors)

# text_emb / sensor_emb shape: (batch_size, 640)
sim = torch.nn.functional.cosine_similarity(text_emb, sensor_emb)
print(f"Cosine similarity: {sim.tolist()}")
```

### Generate Text Conditioned on Sensor Data

```python
prompt = "This sensor reading indicates"
gen_tok = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True)
gen_text = {k: v.to(device) for k, v in gen_tok.items()}

with torch.no_grad():
    output_ids = model.generate(gen_text, sensors, max_new_tokens=50)

for i, ids in enumerate(output_ids):
    print(f"Sample {i}: {tokenizer.decode(ids, skip_special_tokens=True)}")
```

### Get Sensor-Only Embeddings (No Text Needed)

```python
with torch.no_grad():
    sensor_emb = model.get_sensor_embedding(
        input_ids=sensors["input_ids"],
        mask=sensors["attention_mask"],
        time_index=sensors["time_index"],
    )
# sensor_emb shape: (batch_size, 640)
```

### Load Task-Specific Checkpoints

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Download and load a finetuned checkpoint (e.g., HAR)
har_path = hf_hub_download("LeoChen085/SLIP", "har.safetensors")
result = model.load_state_dict(load_file(har_path, device=str(device)), strict=False)
print(f"Loaded HAR checkpoint — missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)}")
```

## Usage

### Pretraining

The pretraining dataset is automatically downloaded from HuggingFace when training starts.

```bash
# 4-GPU distributed pretraining
bash script/train/pretrain.sh
```

Or run directly:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HYDRA_FULL_ERROR=1 \
torchrun --nproc_per_node=4 -m train_main model=slip \
    remark='SLIP_Base'
```

Config overrides can be passed via Hydra CLI (see `config/train.yaml` for defaults: 40 epochs, batch_size=64, lr=2e-4, warmup=4 epochs).

### Linear Probing (11 tasks)

Datasets are automatically downloaded from HuggingFace if not found locally (see `data/lp_dataset/<task_name>/`).

```bash
bash script/eval/lp.sh
```

Or run a single task:
```bash
python -m evaluation.sensor_eval_main \
    --config-name sensor_lp \
    dataset.task_name="wisdm" \
    batch_size=64 \
    finetune="ckpt/SLIP_gemma270.pth" \
    remark="SLIP_Base"
```

### Zero-Shot Evaluation (11 tasks)

Datasets are automatically downloaded from HuggingFace if not found locally (requires `text_y.npy` and `prompt.md` for local use).

```bash
bash script/eval/zs.sh
```

Or run a single task:
```bash
python -m evaluation.zs_eval_main \
    --config-name sensor_zs \
    dataset.task_name="wisdm" \
    batch_size=64 \
    finetune="ckpt/SLIP_gemma270.pth" \
    remark="SLIP_Base"
```

### Supervised Finetuning (QA & Captioning)

SFT datasets are loaded from [LeoChen085/SlipSFTDataset](https://huggingface.co/datasets/LeoChen085/SlipSFTDataset) automatically.

```bash
bash script/eval/sft.sh
```

SFT tasks: `har_cot`, `sleep_cot`, `ecg_cot`, `tsqa`, `m4_caption`

Or run a single task:
```bash
CUDA_VISIBLE_DEVICES=0,1 HYDRA_FULL_ERROR=1 \
torchrun --nproc_per_node=2 -m evaluation.sft \
    dataset.task_name=har_cot \
    finetune='ckpt/SLIP_gemma270.pth' \
    remark='SLIP_SFT' \
    epochs=4 \
    batch_size=64
```

## Dataset Classes

All dataset utilities are in `util/dataset.py`:

| Class | Description |
|-------|-------------|
| `SLIPhfDataset` | Pretraining dataset wrapping a HuggingFace `datasets` split |
| `SLIPCollator` | Collator for pretraining batches (sensor + text) |
| `ZeroshotDataset` | Zero-shot evaluation dataset; auto-downloads from HF if local data is missing |
| `EvalDataset` | Linear probing dataset (train/test split); auto-downloads from HF if local data is missing |
| `EvalCollator` | Collator for linear probing batches |
| `SftDataset` | SFT dataset for QA/captioning tasks; loads from HF via `hf_repo` |
| `SFTCollator` | Collator for SFT batches |

## Checkpoints

All checkpoints are hosted on Hugging Face at [LeoChen085/SLIP](https://huggingface.co/LeoChen085/SLIP). See [Installation Step 4](#4-download-pretrained-checkpoint) for download instructions.

| Checkpoint | Description |
|-----------|-------------|
| `SLIP_gemma270.pth` | Pretrained SLIP base model |
| `har.safetensors` | SFT for HAR chain-of-thought QA |
| `sleep.safetensors` | SFT for Sleep stage chain-of-thought QA |
| `ecg.safetensors` | SFT for ECG-QA chain-of-thought QA |
| `tsqa.safetensors` | SFT for time series QA |
| `caption.safetensors` | SFT for M4 sensor captioning |

## Configuration

All configs use [Hydra](https://hydra.cc/). Key config files:

- **Model**: `config/model/slip.yaml` — Gemma-3-270M backbone, 64 caption queries, split at layer 12, 4 unlocked text encoder layers
- **Sensor Encoder**: `config/sensor_encoder/rope_vit_base.yaml` — 768-dim, 12 layers, 12 heads, full self-attention mode
- **Training**: `config/train.yaml` — AdamW, cosine schedule, gradient clipping (max_norm=1.0)
- **Evaluation**: `config/sensor_lp.yaml`, `config/sensor_zs.yaml`, `config/sft.yaml`

## Citation

```bibtex
@article{chen2026slip,
  title={Learning Transferable Sensor Models via Language-Informed Pretraining},
  author={Chen, Yuliang and Pillai, Arvind and Wu, Yu Yvonne and Griffin, Tess Z. and Marsch, Lisa and Heinz, Michael V. and Jacobson, Nicholas C. and Campbell, Andrew},
  journal={Preprint},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
