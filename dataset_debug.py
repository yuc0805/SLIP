"""
Upload SFT parquet splits to HuggingFace.

    python upload_sft_to_hf.py

Expects the following layout:
    /scratch/leo/data/opentslm_sft_data/{task}/{split}.parquet
"""

import os
from datasets import Dataset, DatasetDict
import pandas as pd
from huggingface_hub import HfApi

HF_REPO = "LeoChen085/SlipSFTDataset"
LOCAL_BASE = "/scratch/leo/data/opentslm_sft_data"
SPLITS = ["train", "test", "val"]


def delete_old_files(api):
    print("Deleting old files from HF repo...")
    files = list(api.list_repo_files(HF_REPO, repo_type="dataset"))
    for f in files:
        # keep README.md, delete everything else
        if f == "README.md":
            continue
        print(f"  Deleting {f}...")
        api.delete_file(path_in_repo=f, repo_id=HF_REPO, repo_type="dataset")
    print("Done cleaning.\n")


def main():
    api = HfApi()
    delete_old_files(api)

    task_dirs = sorted([
        d for d in os.listdir(LOCAL_BASE)
        if os.path.isdir(os.path.join(LOCAL_BASE, d))
    ])
    print(f"Found tasks: {task_dirs}\n")

    for task in task_dirs:
        split_datasets = {}

        for split in SPLITS:
            parquet_path = os.path.join(LOCAL_BASE, task, f"{split}.parquet")
            if not os.path.exists(parquet_path):
                print(f"  [SKIP] {task}/{split}: not found")
                continue

            print(f"  [{task}/{split}] Loading...")
            df = pd.read_parquet(parquet_path)
            df["time_series"] = df["time_series"].apply(
                lambda arrays: [a.tolist() for a in arrays]
            )
            split_datasets[split] = Dataset.from_pandas(df, preserve_index=False)

        if not split_datasets:
            print(f"[SKIP] {task}: no splits found\n")
            continue

        ds_dict = DatasetDict(split_datasets)
        print(f"  Pushing {task} to HF (config_name='{task}')...")
        ds_dict.push_to_hub(HF_REPO, config_name=task)
        print(f"  Done.\n")

    # Verify
    print("=" * 50)
    print("Verification:")
    from datasets import load_dataset
    for task in task_dirs:
        try:
            ds = load_dataset(HF_REPO, name=task)
            for split, sds in ds.items():
                print(f"  [{task}/{split}] OK - {len(sds)} rows, columns: {sds.column_names}")
        except Exception as e:
            print(f"  [{task}] ERROR - {e}")


if __name__ == "__main__":
    main()