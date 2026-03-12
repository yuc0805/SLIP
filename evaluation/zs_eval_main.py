# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
import time
import datetime
from pathlib import Path
import uuid
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import util.misc as misc
from transformers import AutoTokenizer
from util.dataset import SLIPCollator, ZeroshotDataset

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

WANDB_API_KEY = os.getenv("wandb_key")
wandb.login(key=WANDB_API_KEY)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ensure_checkpoint(path, repo="LeoChen085/SLIP"):
    """Download checkpoint from HuggingFace if not found locally."""
    if path and not os.path.exists(path):
        print(f"Checkpoint not found at {path}, downloading from HuggingFace...")
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        safetensors_path = hf_hub_download(repo, "model.safetensors")
        state_dict = load_file(safetensors_path)
        os.makedirs(os.path.dirname(path) or "ckpt", exist_ok=True)
        torch.save({"model": state_dict}, path)
        print(f"Downloaded and converted checkpoint to {path}")


# workspace/HealthSensorSLM-Bench/config/sensor_classification.yaml
@hydra.main(version_base=None, config_path="../config", config_name="sensor_zs.yaml")
def main(cfg: DictConfig):
    ## system setup #############
    misc.init_distributed_mode(cfg)
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False

    # Auto-download checkpoint if needed
    if cfg.finetune:
        ensure_checkpoint(cfg.finetune)

    # prepare dataset
    print('using tokenizer:', cfg.model.llm_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_model_name,use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # ensure pad token is defined
        tokenizer.pad_token = tokenizer.eos_token
        print('pad token not found, set pad token to eos token')

    print('Original vocab size:', tokenizer.vocab_size)

    dataset_name = cfg.dataset.name
    
    if cfg.dataset.fold is not None:
        fold_dict = cfg.dataset.fold

    task_name = cfg.dataset.task_name
    task_config = cfg.dataset.task_config[task_name]

    # setup run name''' build logging information'''
    if cfg.remark is None:
        remark = f"{cfg.sensor_encoder.name}_{dataset_name}_{task_name}"
    else:
        remark = f"{cfg.remark}_{cfg.sensor_encoder.name}_{dataset_name}_{task_name}"

    print('model remark is', remark)

    ### ### ###
    # prepare logging.
    unique_id = uuid.uuid4().hex[:8] 
    root_output_dir = cfg.output_dir
    root_log_dir = cfg.log_dir
    # Begin k-fold.
    global_acc = []

    
    return_patch = cfg.dataset.get('return_patch', True)
    
    kfold = 1
    use_fixed_patch = cfg.dataset.patch_size
    
    for fold_idx in range(kfold): 
        print('running fold', fold_idx)

        is_normalize = task_config['is_normalize']
        ps = task_config.get('patch_size', 16)
        collate_fn = SLIPCollator(tokenizer,max_len=1536)
        

        cfg.dataset.is_normalize = is_normalize # just for the wandb log
        cfg.dataset.patch_size = ps # just for the wandb log

        print('is_normalize:', is_normalize)
        print('Using Patch Size:', ps)

        dataset_val = ZeroshotDataset(
            data_dir = cfg.dataset.data_dir,
            sensor_aug = False,
            is_normalize = is_normalize,
            patch_size = ps,
            return_patch = return_patch,
        )

        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        cfg.log_dir = os.path.join(root_log_dir, remark, timestamp)
        cfg.output_dir = os.path.join(root_output_dir, remark, timestamp)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        cfg.linear_probe = 'zero_shot'
        if cfg.log_dir is not None and not cfg.eval:
            log_writer = wandb.init(project='Sensor-Classification',
                                    config=OmegaConf.to_container(cfg, resolve=True),
                                    dir=cfg.log_dir,
                                    group=f'{remark}_{task_name}_{unique_id}',
                                    name=f"fold_{fold_idx}",
                                    reinit=True)
        else:
            log_writer = None

        print('Using dataset',dataset_name)
        print("Number of Testing Samples:", len(dataset_val))
    

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Loading Model

        model = instantiate(cfg.model, tokenizer=tokenizer)
        if cfg.finetune and not cfg.eval:
            print("Load pre-trained checkpoint from: %s" % cfg.finetune)
            checkpoint = torch.load(cfg.finetune, map_location='cpu',weights_only=False)
            checkpoint_model = checkpoint['model']
            
            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)
        
        model.to(device)
        
        print("Model = %s" % str(model))
        n_total_parameters = sum(p.numel() for p in model.parameters())
        print('Total number of parameters: %.2f' % (n_total_parameters))


        start_time = time.time()
        test_stats = zs_eval(model, data_loader_val, device=device)           


        if log_writer is not None:
            log_writer.log({f"perf/test_{k}": v for k, v in test_stats.items()})

        max_accuracy = test_stats['acc1']
        global_acc.append(max_accuracy)
        
    # find the mean, std of the global_acc
    print(f'Global accuracy:, {np.mean(global_acc)}+- {np.std(global_acc)}')
    log_writer.log({'kfold-mean': np.mean(global_acc),
                    'kfold-std': np.std(global_acc)})
    

    wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    return max_accuracy



import torch.nn.functional as F
def zs_eval(model, dataloader, device='cuda'):
    # similarity: (N_samples, N_classes)
    # all_caption: list of strings (len = N_samples)
    similarity, all_caption, all_sensor, sensor_feat, text_feat = retrival_eval(model, dataloader)

    # 1. Create a mapping from string labels to integers based on appearance order
    # This preserves the link between the similarity matrix columns and the class identity
    unique_labels_list = []
    label_to_idx = {}
    for cap in all_caption:
        if cap not in label_to_idx:
            label_to_idx[cap] = len(unique_labels_list)
            unique_labels_list.append(cap)
    
    # 2. Convert ground truth strings to numerical indices
    y_true = np.array([label_to_idx[cap] for cap in all_caption])
    
    # 3. Handle Predictions
    # similarity.argmax(dim=1) gives the index of the matching caption/column
    pred_indices = similarity.argmax(dim=1).cpu().numpy()
    
    # Map the predicted index back to a string, then to our stable integer ID
    # This is the "safe" way to handle cases where similarity columns != all_caption indices
    y_pred = np.array([label_to_idx[all_caption[i]] for i in pred_indices])

    # 4. Probabilities for AUC-ROC
    # We apply softmax to get scores in [0, 1]
    probs = F.softmax(similarity, dim=1).cpu().numpy()

    # --- Metric Calculations ---

    # Accuracy (Matches your original logic: pred_string == true_string)
    acc1 = (y_pred == y_true).mean() * 100

    # Balanced Accuracy: Average of recall for each class
    balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100

    # F1 Score: Macro-averaged to treat each class equally
    f1 = f1_score(y_true, y_pred, average='macro') * 100


    print(f"Zero-Shot Results:")
    print(f"  Acc: {acc1:.2f}% | BalAcc: {balanced_acc:.2f}% | F1: {f1:.2f}%")

    return {
        "acc1": acc1,
        "balanced_acc": balanced_acc,
        "f1": f1
    }


def retrival_eval(model, dataloader):
    model = model.eval()
    
    sensor_feat = []
    text_feat = []

    all_caption = []
    all_sensor = []

    for batch in tqdm(dataloader):
        text_input = batch['text']
        sensor_input = batch['sensor']

        caption = batch['text']['description']  # list of strings
        all_caption.extend(caption)
        all_sensor.extend(sensor_input['input_ids'])

        # move to cuda
        text_input = {
                k: (v.to('cuda', non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in text_input.items()
            }
        sensor_input = {
                k: (v.to('cuda', non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in sensor_input.items()
            }
    
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                text_cls, img_cls = model.get_embedding(
                    text_input, sensor_input)
                
        sensor_feat.append(img_cls.cpu())
        text_feat.append(text_cls.cpu())

        # compute similarity
    sensor_feat = torch.cat(sensor_feat, dim=0)
    text_feat = torch.cat(text_feat, dim=0)

    # compute cosine similarity
    similarity = sensor_feat @ text_feat.T
    
    return similarity, all_caption, all_sensor,sensor_feat, text_feat

if __name__ == '__main__':
    main()

'''
CUDA_VISIBLE_DEVICES=7 \
 python -m evaluation.zs_eval_main \
        --config-name sensor_zs \
        dataset.task_name="PPG_CVA" \
        batch_size=16 \
        finetune="ckpt/SLIP_gemma270.pth" \
        remark="SLIP_Base"

'''