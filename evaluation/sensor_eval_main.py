# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import os
import time
import datetime
from pathlib import Path
import uuid
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from transformers import AutoProcessor,AutoTokenizer
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.dataset import EvalDataset, EvalCollator
from util.head import ClassificationWrapper
import util.lr_decay as lrd
from evaluation.engine_eval import train_one_epoch, evaluate

WANDB_API_KEY = os.getenv("wandb_key")
wandb.login(key=WANDB_API_KEY)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ensure_checkpoint(path, repo="LeoChen085/SLIP"):
    """Download checkpoint from HuggingFace if not found locally.

    The HF repo stores the base model as model.safetensors (flat state dict).
    This function downloads it, wraps it as {'model': state_dict}, and saves
    as a .pth file so the rest of the code can use torch.load(path)['model'].
    """
    if path and not os.path.exists(path):
        print(f"Checkpoint not found at {path}, downloading from HuggingFace...")
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        safetensors_path = hf_hub_download(repo, "model.safetensors")
        state_dict = load_file(safetensors_path)
        os.makedirs(os.path.dirname(path) or "ckpt", exist_ok=True)
        torch.save({"model": state_dict}, path)
        print(f"Downloaded and converted checkpoint to {path}")


def load_eval_datasets(cfg, task_name, task_config, return_patch):
    """Load eval datasets from local files or HuggingFace."""
    is_normalize = task_config['is_normalize']
    ps = cfg.dataset.patch_size if cfg.dataset.patch_size else task_config.get('patch_size', 16)

    data_dir = cfg.dataset.data_dir
    hf_repo = cfg.dataset.get('hf_repo', 'LeoChen085/SlipDataset')

    dataset_train = EvalDataset(
        data_dir=data_dir,
        is_train=True,
        is_normalize=is_normalize,
        patch_size=ps,
        return_patch=return_patch,
        hf_repo=hf_repo,
    )
    dataset_val = EvalDataset(
        data_dir=data_dir,
        is_train=False,
        is_normalize=is_normalize,
        patch_size=ps,
        return_patch=return_patch,
        hf_repo=hf_repo,
    )

    return dataset_train, dataset_val, ps, is_normalize

# workspace/HealthSensorSLM-Bench/config/sensor_classification.yaml
@hydra.main(version_base=None, config_path="../config", config_name="sensor_classification.yaml")
def main(cfg: DictConfig):
    ## system setup #############
    misc.init_distributed_mode(cfg)
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False


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
    ## prepare hyperparameters 
    eff_batch_size = cfg.batch_size * cfg.accum_iter * misc.get_world_size()
    if cfg.lr is None:  # only base_lr is specified
        cfg.lr = cfg.blr * eff_batch_size / 256

    print("base lr: %.2e" % (cfg.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % cfg.lr)

    print("accumulate grad iterations: %d" % cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    print("lr: %.3e" % cfg.lr)
    ### ### ###
    # prepare logging.
    unique_id = uuid.uuid4().hex[:8] 
    root_output_dir = cfg.output_dir
    root_log_dir = cfg.log_dir
    # Begin k-fold.
    global_acc = []

    return_patch = cfg.dataset.get('return_patch', True)

    # Auto-download checkpoint if needed
    if cfg.finetune:
        ensure_checkpoint(cfg.finetune)

    for fold_idx, (fold_key, split) in enumerate(fold_dict.items()):
        print('running fold', fold_idx, fold_key)

        dataset_train, dataset_val, ps, is_normalize = load_eval_datasets(
            cfg, task_name, task_config, return_patch
        )

        print('is_normalize:', is_normalize)
        print('Using Patch Size:', ps)

        cfg.dataset.is_normalize = is_normalize # just for the wandb log
        cfg.dataset.patch_size = ps # just for the wandb log


        global_rank = 0
        if cfg.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            if cfg.dist_eval:
                sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        cfg.log_dir = os.path.join(root_log_dir, remark, timestamp)
        cfg.output_dir = os.path.join(root_output_dir, remark, timestamp)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        if cfg.log_dir is not None and global_rank == 0 and not cfg.eval:
            log_writer = wandb.init(project='Sensor-Classification',
                                    config=OmegaConf.to_container(cfg, resolve=True),
                                    dir=cfg.log_dir,
                                    group=f'{remark}_{task_name}_{unique_id}',
                                    name=f"fold_{fold_idx}",
                                    reinit=True)
        else:
            log_writer = None

        print('Using dataset',dataset_name)
        print("Number of Training Samples:", len(dataset_train))
        print("Number of Testing Samples:", len(dataset_val))
        
        collate_fn = lambda batch: EvalCollator(batch, return_patch=return_patch)
            
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            drop_last=True,
            collate_fn=collate_fn,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Loading Model
        model_flag = cfg.get('model', False)
        if model_flag:
            print('using attention pooling feature extractor!')
            feature_extractor = instantiate(model_flag, tokenizer=None)
            encoder_prefix = ''
            feature_extractor.forward = feature_extractor.embed_sensor
            glob_pool = 'cls'
            model_flag = True

        else:   
            encoder_prefix = cfg.get('encoder_prefix', 'sensor_encoder.') # for our vit
            feature_extractor = instantiate(cfg.sensor_encoder)
            glob_pool = 'avg'
            model_flag = False
        

        if cfg.finetune and not cfg.eval:
            checkpoint = torch.load(cfg.finetune, map_location='cpu',weights_only=False)

            print("Load pre-trained checkpoint from: %s" % cfg.finetune)
            checkpoint_model = checkpoint['model']
            encoder_weights = load_encoder_weights(checkpoint_model,encoder_prefix=encoder_prefix)
            
            # load pre-trained model
            msg = feature_extractor.load_state_dict(encoder_weights, strict=False)
            print(msg)
        
        feature_extractor.to(device)
        print('Pooling type:', glob_pool)
        model = ClassificationWrapper(feature_extractor,
                                      head_type=task_config.head_type,
                                      num_classes=task_config.num_classes,
                                      y_range=task_config.get('y_range', None),
                                      glob_pool = glob_pool)
        
        if cfg.linear_probe:
            # hack: revise model's head with BN
            model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
            # freeze all but the head
            for _, p in model.named_parameters():
                p.requires_grad = False
            for _, p in model.head.named_parameters():
                p.requires_grad = True

            # unfreeze the attn_pooling
            if model_flag:
                for _, p in model.model.img_attn_pool.named_parameters():
                    p.requires_grad = True
            

        model.to(device)

        model_without_ddp = model
        
        print("Model = %s" % str(model))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of training params : %.2f' % (n_parameters))

        n_total_parameters = sum(p.numel() for p in model.parameters())
        print('Total number of parameters: %.2f' % (n_total_parameters))

        if cfg.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
            model_without_ddp = model.module

        optimizer =  torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            )

        print(optimizer)

        loss_scaler = NativeScaler()
        
        if task_config.head_type == 'classification':
            # w = dataset_train.make_ce_weights().float().to(device)
            # print('Class weights:', w)
            criterion = torch.nn.CrossEntropyLoss()
            metric_key = 'acc1'
            max_accuracy = 0.0
        else:
            criterion = torch.nn.MSELoss()
            metric_key = 'mae'
            max_accuracy = 100.0 # some extreme large value
            
        print(f"Start training for {cfg.epochs} epochs")
        start_time = time.time()
        best_metric = {}
        for epoch in range(cfg.start_epoch, cfg.epochs):
            if cfg.distributed: 
                data_loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, epoch, loss_scaler,
                max_norm=cfg.max_norm, 
                log_writer=log_writer,
                args=cfg, device=device,
            )
            if cfg.output_dir and (epoch + 1 == cfg.epochs): #
                misc.save_model(
                    args=cfg, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

            test_stats = evaluate(data_loader_val, model, criterion, device)

            
            current_metric = test_stats[metric_key]
            print(f"{metric_key} of the network on the {len(dataset_val)} test images: {current_metric:.1f}%")

            improved = (
                current_metric > max_accuracy if metric_key == "acc1"
                else current_metric < max_accuracy
            )

            if improved:
                max_accuracy = current_metric
                best_metric.update(epoch=epoch, **{metric_key: max_accuracy})

                if cfg.output_dir:
                    misc.save_model(
                        args=cfg, model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler, epoch="best"
                    )

            print(f'Best {metric_key}: {max_accuracy:.2f}%')

            # save final epoch checkpoint
            if cfg.output_dir and epoch + 1 == cfg.epochs:
                misc.save_model(
                    args=cfg, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
                )

            if log_writer is not None:
                log_writer.log({f"perf/test_{k}": v for k, v in test_stats.items()})

        global_acc.append(max_accuracy)
        if log_writer is not None:
            log_writer.log({f"best/best_{k}": v for k, v in best_metric.items()})
        
    # find the mean, std of the global_acc
    print(f'Global accuracy:, {np.mean(global_acc)}+- {np.std(global_acc)}')
    log_writer.log({'kfold-mean': np.mean(global_acc),
                    'kfold-std': np.std(global_acc)})
    

    wandb.finish()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    return max_accuracy

def load_encoder_weights(checkpoint_model, encoder_prefix="img_encoder."):
    # check if any keys start with the prefix
    has_prefix = any(k.startswith(encoder_prefix) for k in checkpoint_model.keys())

    if has_prefix:
        encoder_weights = {
            k[len(encoder_prefix):]: v
            for k, v in checkpoint_model.items()
            if k.startswith(encoder_prefix)
        }
    else:
        # no prefix in keys — just copy them directly
        encoder_weights = checkpoint_model

    return encoder_weights

if __name__ == '__main__':
    main()


    '''
    example run:

    python -m evaluation.sensor_eval_main \
    --config-name sensor_lp \
    dataset.task_name="PPG_CVA" \
    batch_size=64 \
    finetune="ckpt/SLIP_gemma270.pth" \
    remark="SLIP_Base"
    
    '''

