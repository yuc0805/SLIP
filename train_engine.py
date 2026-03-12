# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import json
import math
import sys
from typing import Iterable
import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    max_norm:float=1.0,
                    log_writer=None,
                    args=None):
    model.train(True)

    if args.is_sft:
        core = model.module if hasattr(model, "module") else model


    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.dir))

    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        sensor = data_dict['sensor']
        text = data_dict['text']
        BS = len(text)


        if sensor:
            sensor = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in sensor.items()
            }

        text = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in text.items()
        }
    
        if 'prompt' in data_dict:
            prompt = {k: v.to(device, non_blocking=True) for k, v in data_dict['prompt'].items()}
        else:
            prompt = None

        
        if args.is_sft:
            # SFT ...
            if args.model.name == 'slip':
                # SLIP training engine.
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss_dict = core.sft_training(text, sensor)
            else:
                # LLM finetuning engine. i.e. CHATTS
                loss = model(**text).loss
                loss_dict = {'loss':loss}
        else:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss_dict = model(text, sensor, prompt)

        loss = loss_dict['loss'].clone()
        loss_value = loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            
            sys.exit(1)

        loss /= accum_iter
        #........................................

        loss.backward()

        if (data_iter_step + 1) % accum_iter == 0:
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()
        # ....................................

        torch.cuda.synchronize()

        step_metrics = {
            k: float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in loss_dict.items()
        }
        metric_logger.update(**step_metrics)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        reduced = {k: misc.all_reduce_mean(v) for k, v in step_metrics.items()}
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            log_writer.log({f'train_{k}': v for k, v in reduced.items()})
            log_writer.log({'lr': lr})
            log_writer.log({'batch_size': BS})



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@ torch.no_grad()
def validation_evaluate(model: torch.nn.Module,
                        data_loader: Iterable,
                        device: torch.device,
                        log_writer=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'

    with torch.no_grad():
        for data_dict in metric_logger.log_every(data_loader, 50, header):
            sensor = data_dict['sensor']
            text = data_dict['text']

            
            sensor = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in sensor.items()
            }
            text = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in text.items()
            }

            if 'prompt' in data_dict:
                prompt = {k: v.to(device, non_blocking=True) for k, v in data_dict['prompt'].items()}
            else:
                prompt = None

            with torch.cuda.amp.autocast():
                loss_dict = model(text, sensor, prompt)

            step_metrics = {
                k: float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
                for k, v in loss_dict.items()
            }
            metric_logger.update(**step_metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    print("Averaged stats:", metric_logger)

    if log_writer is not None:
        log_writer.log({f'val_{k}': meter.global_avg for k, meter in metric_logger.meters.items()})

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


from tqdm import tqdm
import torch.nn.functional as F
def evaluate_sft(model, data_loader, device, log_writer=None):
    model.eval()
    core = model.module if hasattr(model, "module") else model

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Validation:"

    results = []
    with torch.no_grad():
        for data_dict in metric_logger.log_every(data_loader, 50, header):
            sensor = data_dict["sensor"]
            text = data_dict["text"]

            if sensor:
                sensor = {
                    k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in sensor.items()
                }
            text = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in text.items()
            }

            if sensor:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss_dict = core.sft_training(text,sensor,)
            else:
                loss = model(**text).loss
                loss_dict = {'loss':loss}

            metric_logger.update(loss=loss_dict['loss'])
            
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)

    if log_writer is not None:
        log_writer.log(
            {f"val_{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
        )

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return results

            
                    



