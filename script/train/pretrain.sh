#!/bin/bash
# SLIP Pretraining
# Dataset is automatically downloaded from HuggingFace: LeoChen085/SlipDataset

CUDA_VISIBLE_DEVICES=0,1,2,3 HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=4 -m train_main model=slip \
    remark='SLIP_Base'
