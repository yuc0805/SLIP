# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
from timm.utils import accuracy
import util.misc as misc
import util.lr_sched as lr_sched
from util.metrics import ClassificationMetrics, RegressionMetrics


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 1.0,
                    log_writer=None, device = torch.device,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        is_classification = True
    else:
        is_classification = False

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.dir))

    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        data_dict = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in data_dict.items()
        }

        targets = data_dict['targets']
        samples = data_dict['samples']
        mask = data_dict['mask']
        time_index = data_dict['time_index']

        with torch.amp.autocast("cuda", enabled=False):
            outputs = model(samples, mask,time_index) # bs x num_classes
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        batch_size = len(samples)
        
        if is_classification:
            acc1, _ = accuracy(outputs, targets, topk=(1, 3))
            metric_logger.meters['train_acc1'].update(acc1.item(), n=batch_size)
        else:
            # mean absolute error
            mae = torch.abs(outputs - targets).mean()
            metric_logger.meters['train_mae'].update(mae.item(), n=batch_size)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize() 

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if is_classification:
            acc1_reduce = misc.all_reduce_mean(acc1)
            train_stats = {
                'loss': loss_value_reduce,
                'train_acc1': acc1_reduce,
                'lr': max_lr
            }
        else:
            mae_reduce = misc.all_reduce_mean(mae)
            train_stats = {
                'loss': loss_value_reduce,
                'train_mae': mae_reduce,
                'lr': max_lr
            }



        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.log(train_stats, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, criterion, device):
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        is_classification = True
    else:
        is_classification = False
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if is_classification:
        metrics = ClassificationMetrics(num_classes=model.num_classes, 
                                        device=device)
    else:
        metrics = RegressionMetrics(device=device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        target = batch['targets']
        sensor = batch['samples']
        mask = batch['mask']
        time_index = batch['time_index']
        
        output = model(sensor, mask, time_index)
        loss = criterion(output, target)

        metrics.update(output, target)

        metric_logger.update(loss=loss.item())
        # batch_size = images.shape[0]
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    results = metrics.compute()

    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update(results)

    for k, v in return_dict.items():
        print(f"{k}: {v:.4f}")

    return return_dict


##
def probe_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 1.0,
                    log_writer=None, device = torch.device,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        is_classification = True
    else:
        is_classification = False

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.dir))

    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        data_dict = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in data_dict.items()
        }

        targets = data_dict['targets']
        samples = data_dict['samples']

        with torch.amp.autocast("cuda", enabled=False):
            outputs = model(samples) # bs x num_classes
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        batch_size = len(samples)
        
        if is_classification:
            acc1, _ = accuracy(outputs, targets, topk=(1, 3))
            metric_logger.meters['train_acc1'].update(acc1.item(), n=batch_size)
        else:
            # mean absolute error
            mae = torch.abs(outputs - targets).mean()
            metric_logger.meters['train_mae'].update(mae.item(), n=batch_size)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize() 

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if is_classification:
            acc1_reduce = misc.all_reduce_mean(acc1)
            train_stats = {
                'loss': loss_value_reduce,
                'train_acc1': acc1_reduce,
                'lr': max_lr
            }
        else:
            mae_reduce = misc.all_reduce_mean(mae)
            train_stats = {
                'loss': loss_value_reduce,
                'train_mae': mae_reduce,
                'lr': max_lr
            }



        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.log(train_stats, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def probe_evaluate(data_loader, model, criterion, num_classes, device,):
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        is_classification = True
    else:
        is_classification = False
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if is_classification:
        metrics = ClassificationMetrics(num_classes=num_classes, 
                                        device=device)
    else:
        metrics = RegressionMetrics(device=device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        target = batch['targets']
        sensor = batch['samples']
        
        output = model(sensor)
        loss = criterion(output, target)

        metrics.update(output, target)

        metric_logger.update(loss=loss.item())
        # batch_size = images.shape[0]
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    results = metrics.compute()

    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update(results)

    for k, v in return_dict.items():
        print(f"{k}: {v:.4f}")

    return return_dict

