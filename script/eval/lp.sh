# !/bin/bash
# Linear Probing Evaluation (11 tasks)
#
# Checkpoint and datasets are automatically downloaded from HuggingFace if not found locally.
# To use local data instead, place files under data/lp_dataset/<task_name>/

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

    python -m evaluation.sensor_eval_main \
        --config-name sensor_lp \
        dataset.task_name="$task" \
        batch_size=${batch_sizes[$task]} \
        finetune="ckpt/slip.pth" \
        remark="SLIP_Base"
done
