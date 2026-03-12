# !/bin/bash
# Zero-Shot Evaluation (11 tasks)
#
# Checkpoint is automatically downloaded from HuggingFace if not found locally.
# Zero-shot datasets must be placed under data/zs_dataset/<task_name>/ (requires text_y.npy and prompt.md).

tasks=(
    "wisdm"
    "uci_har"
    "wesad"
    "PPG_CVA"
    "PPG_DM"
    "PPG_HTN"
    "studentlife"
    "ptbxl"
    "sleepEDF"
    "AsphaltObstacles"
    "Beijing_AQI"
)

declare -A batch_sizes=(
    ["wisdm"]=64
    ["uci_har"]=128
    ["wesad"]=32
    ["PPG_CVA"]=16
    ["PPG_DM"]=16
    ["PPG_HTN"]=16
    ['studentlife']=16
    ['ptbxl']=64
    ["sleepEDF"]=128
    ["AsphaltObstacles"]=32
    ["Beijing_AQI"]=32
)


for task in "${tasks[@]}"; do
    echo "Running task: $task"

    python -m evaluation.zs_eval_main \
        --config-name sensor_zs \
        dataset.task_name="$task" \
        batch_size=${batch_sizes[$task]} \
        finetune="ckpt/SLIP_gemma270.pth" \
        remark="SLIP_Base"
done
