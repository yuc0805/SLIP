#!/bin/bash
# Supervised Finetuning (QA & Captioning)
#
# Checkpoint is automatically downloaded from HuggingFace if not found locally.
# SFT datasets must be placed under data/opentslm_sft_data/<task_name>/.

tasks=(
    "har_cot"
    "sleep_cot"
    "ecg_cot"
    "tsqa"
)

declare -A batch_size=(
    ["har_cot"]=64
    ["sleep_cot"]=32
    ["ecg_cot"]=32
    ["tsqa"]=64
)

declare -A epochs=(
    ["har_cot"]=4
    ["sleep_cot"]=10 # match the iteration of other dataset loosely.
    ["ecg_cot"]=4
    ["tsqa"]=4
)


for task in "${tasks[@]}"; do
    echo "Running task: $task"

    CUDA_VISIBLE_DEVICES=0,1 \
    HYDRA_FULL_ERROR=1 \
    torchrun --nproc_per_node=2 -m evaluation.sft \
        dataset.task_name=$task \
        finetune='ckpt/SLIP_gemma270.pth' \
        remark='SLIP_SFT' \
        epochs=${epochs[$task]} \
        batch_size=${batch_size[$task]}
done

# M4 Captioning
# This will output caption.json file, need to do post-hoc evaluation...
CUDA_VISIBLE_DEVICES=0,1 \
HYDRA_FULL_ERROR=1 \
torchrun --nproc_per_node=2 -m evaluation.sft \
    dataset.task_name=m4_caption \
    finetune='ckpt/SLIP_gemma270.pth' \
    epochs=4 \
    remark='SLIP_SFT'
