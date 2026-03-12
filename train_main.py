# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import os
import random
import time
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from util.dataset import CLIPDataset, CLIPCollator,WISDM_Caption
# from util.sampler import TokenSampler
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from train_engine import train_one_epoch, validation_evaluate
from transformers import AutoTokenizer 


@hydra.main(version_base=None, config_path="config", config_name="train.yaml")
def main(cfg: DictConfig):
    # prepare some training parameters
    if cfg.save_every_epoch is None:
        cfg.save_every_epoch = int(0.1*cfg.epochs)

    ## system setup #############
    misc.init_distributed_mode(cfg)
    print(OmegaConf.to_yaml(cfg))

    print('======================= starting pretrain =======================')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    device = torch.device(cfg.device)
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_model_name,use_fast=True)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # ensure pad token is defined
        tokenizer.pad_token = tokenizer.eos_token
        print('pad token not found, set pad token to eos token')

    print('Original vocab size:', tokenizer.vocab_size)

    # Load dataset from HuggingFace if hf_repo is specified
    if cfg.dataset.get("hf_repo", None):
        print(f"Loading dataset from HuggingFace repo: {cfg.dataset.hf_repo}")
        from datasets import load_dataset as load_hf_dataset
        from huggingface_hub import hf_hub_download
        hf_ds = load_hf_dataset(cfg.dataset.hf_repo)
        meta_path = hf_hub_download(
            repo_id=cfg.dataset.hf_repo,
            filename=cfg.dataset.get("hf_meta_file", "meta.csv"),
            repo_type="dataset",
        )
        dataset_train = instantiate(
            cfg.dataset.dataset,
            hf_dataset=hf_ds["train"],
            meta_info=meta_path,
        )
    else:
        dataset_train = instantiate(cfg.dataset.dataset)
    collate_fn = instantiate(cfg.dataset.collator,
                             tokenizer=tokenizer,)


    global_rank = 0
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )


        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if cfg.remark is None:
        cfg.remark = f'{cfg.model.name}_{cfg.dataset.name}'
    else:
        cfg.remark = f'{cfg.remark}_{cfg.model.name}_{cfg.dataset.name}'
    print('remark: ', cfg.remark)

    cfg.log_dir = os.path.join(cfg.log_dir,cfg.remark,timestamp)
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.remark, timestamp)
    os.makedirs(cfg.output_dir, exist_ok=True)
    if global_rank == 0 and cfg.log_dir is not None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = wandb.init(
            project='HealthSLM-Alignment-Train',  # Specify your project
            config= OmegaConf.to_container(cfg, resolve=True),
            dir=cfg.log_dir,
            name=cfg.remark,)
    else:
        log_writer = None
        
    print("Number of Samples:", len(dataset_train))
    cfg.batch_size = min(cfg.batch_size, len(dataset_train) // misc.get_world_size())
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        # batch_sampler=sampler_train,
        # drop_last=False,
        sampler=sampler_train,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        collate_fn = collate_fn,
        prefetch_factor=cfg.prefetch_factor,
    )
    
    model = instantiate(cfg.model,tokenizer=tokenizer)
    

    if cfg.finetune:
        # SFT ...
        print('SFT finetune from %s' % cfg.finetune)
        checkpoint = torch.load(cfg.finetune, map_location='cpu',weights_only=False)
        checkpoint_model = checkpoint['model']
        
        # extract encoder weights
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        cfg.is_sft=True
    
    
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = cfg.batch_size * cfg.accum_iter * misc.get_world_size()
    
    if cfg.lr is None:  # only base_lr is specified
        cfg.lr = cfg.blr * eff_batch_size / 256

    print("base lr: %.2e" % (cfg.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % cfg.lr)

    print("accumulate grad iterations: %d" % cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    print("lr: %.3e" % cfg.lr)

    if cfg.distributed:
        print('use distributed!!')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=False,static_graph=True)
        # model.module = torch.compile(model.module)
        model_without_ddp = model.module
    
    # Define Optimizer
    
    if len(model_without_ddp.get_lora_parameters())==0:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            betas=cfg.betas,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            )


    else:
        # 1. Capture all trainable parameters
        lora_params = []
        rest_params = []

        for name, param in model_without_ddp.named_parameters():
            if not param.requires_grad:
                continue
            
            # True LoRA adapters have "lora_A" or "lora_B" in the name
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
            else:
                # This includes Cross-Attn, Tokens, Head, and Bridge layers
                rest_params.append(param)

        # 3. Calculation for printing
        def count_m(params): return sum(p.numel() for p in params) / 1e6

        total_m = sum(p.numel() for p in model_without_ddp.parameters()) / 1e6
        lora_m = count_m(lora_params)
        rest_m = count_m(rest_params)
        trainable_m = lora_m + rest_m
        frozen_m = total_m - trainable_m

        print("\n" + "="*60)
        print("FIXED OPTIMIZER PARAMETER BREAKDOWN")
        print("="*60)
        print(f"Total Parameters:          {total_m:8.2f}M")
        print(f"Total Trainable:           {trainable_m:8.2f}M ({(trainable_m/total_m)*100:.1f}%)")
        print("-" * 60)
        # These are the ones that SHOULD be large if you unfreeze tokens
        print(f"Full-Parameter (rest):     {rest_m:8.2f}M  <- Cross-Attn, Tokens, Bridge")
        # These SHOULD be small (Rank matrices)
        print(f"LoRA Adapters:             {lora_m:8.2f}M  <- Only A and B matrices")
        print("-" * 60)
        print(f"Frozen Saved:              {frozen_m:8.2f}M")
        print("="*60 + "\n")

        # 4. Optimizer setup
        lora_lr = cfg.lora_lr if cfg.lora_lr is not None else cfg.lr * 0.1
        param_groups = [
            {"params": rest_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": lora_params, "lr": lora_lr, "weight_decay": cfg.weight_decay}
        ]
        optimizer = torch.optim.AdamW(param_groups, betas=cfg.betas, eps=1e-8)
    


    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=cfg, model_without_ddp=model_without_ddp,
                     optimizer=optimizer, loss_scaler=loss_scaler)

    num_params = sum(p.numel() for p in model.parameters())
    print('number of params: %.2fM' % (num_params / 1.e6))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable number of params: %.2fM' % (num_params / 1.e6))

    print(f"Start training for {cfg.epochs} epochs")
    start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.epochs):
        try:
            dataset_train.resample_epoch()
        except:
            if cfg.distributed:
                data_loader_train.sampler.set_epoch(epoch)
           # data_loader_train.batch_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=cfg.max_norm,
            log_writer=log_writer,
            args=cfg)

        # val_stats = validation_evaluate(model, data_loader_val, device,log_writer)

        if cfg.output_dir and (epoch % cfg.save_every_epoch == 0 or epoch + 1 == cfg.epochs):
            misc.save_model(
                args=cfg, model=model, 
                model_without_ddp=model_without_ddp, 
                optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

            


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()

''''
CUDA_VISIBLE_DEVICES=0,1,2,3 HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=4 -m train_main model=slip \
    remark='SLIP_Base'

# Detail config can be find in config/tarin.yaml and config/model/slip.yaml
'''