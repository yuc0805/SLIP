# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import string
import uuid
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import time
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from train_engine import train_one_epoch,evaluate_sft

from transformers import AutoProcessor, AutoTokenizer 
from util.dataset import SftDataset, SFTCollator, SFTTensorCollator
import torch.distributed as dist
from transformers import AutoModelForCausalLM

@hydra.main(version_base=None, config_path="../config", config_name="sft.yaml")
def main(cfg: DictConfig):
    # prepare some training parameters
    if cfg.save_every_epoch is None:
        save_every_epoch = int(0.1*cfg.epochs)
        cfg.save_every_epoch = max(1, save_every_epoch)

    ## system setup #############
    misc.init_distributed_mode(cfg)
    print(OmegaConf.to_yaml(cfg))

    task_name = cfg.dataset.task_name
    task_config = cfg.dataset.task_config[task_name]
    print('======================= starting pretrain =======================')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    device = torch.device(cfg.device)
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    ### ### ###
    # prepare logging.
    unique_id = uuid.uuid4().hex[:8] 
    root_output_dir = cfg.output_dir
    root_log_dir = cfg.log_dir
    
    # prepare dataset
    print('using tokenizer:', cfg.model.llm_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_model_name,use_fast=True)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # ensure pad token is defined
        tokenizer.pad_token = tokenizer.eos_token
        print('pad token not found, set pad token to eos token')

    print('Original vocab size:', tokenizer.vocab_size)

    is_normalize = task_config.get('is_normalize',True)
    if cfg.dataset.patch_size:
        ps = cfg.dataset.patch_size
    else:
        ps = task_config.get('patch_size', 16)

    print('is_normalize:', is_normalize)
    print('Using Patch Size:', ps)



    if cfg.model.name == 'slip':  
        dataset_train = SftDataset(data_dir=cfg.dataset.data_dir,split='train',
                                is_normalize=is_normalize,patch_size=ps,)
        dataset_val = SftDataset(data_dir=cfg.dataset.data_dir,split='val',
                                    is_normalize=is_normalize,patch_size=ps,
                                    sensor_aug=False)
        dataset_test = SftDataset(data_dir=cfg.dataset.data_dir,split='test',
                                    is_normalize=is_normalize,patch_size=ps,
                                    sensor_aug=False)
        
    else:
        # LLM Branch,  e.g., ChatTS
        from util.dataset import ChatTsSftDataset
        dataset_train = ChatTsSftDataset(
            data_dir=cfg.dataset.data_dir,
            split='train',
            input_mode=cfg.model.input_mode
        )
        dataset_val = ChatTsSftDataset(
            data_dir=cfg.dataset.data_dir,
            split='val',
            sensor_aug=False,
            input_mode=cfg.model.input_mode
        )
        dataset_test = ChatTsSftDataset(
            data_dir=cfg.dataset.data_dir,
            split='test',
            sensor_aug=False,
            input_mode=cfg.model.input_mode
        )

    if cfg.sensor_encoder.patch_size == 16:
        collate_fn = SFTTensorCollator(tokenizer=tokenizer, 
                                max_len=2880) # use longer for SFT (CoT)
    
    elif cfg.model.name == 'slip':
        collate_fn = SFTCollator(tokenizer=tokenizer, 
                                max_len=2880)
        # test_collate_fn = SFTCollator(tokenizer=tokenizer, 
        #                         max_len=2880,
        #                         is_test=True)
        test_collate_fn = SFTCollator(tokenizer=tokenizer, 
                                max_len=2880,is_test=True)

    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_model_name, trust_remote_code=True)
        tokenizer.padding_side = 'left'

        if cfg.model.input_mode == 'ts':
            processor = AutoProcessor.from_pretrained(cfg.model.llm_model_name, trust_remote_code=True, tokenizer=tokenizer)
        else:
            print('TS as Text, only loading tokenizer')
            processor = tokenizer

        from util.dataset import ChatTsSftcollator
        collate_fn = lambda batch: ChatTsSftcollator(processor, 
                                                     batch, 
                                                     input_mode=cfg.model.input_mode)
        test_collate_fn = lambda batch: ChatTsSftcollator(processor,
                                                     batch, 
                                                     input_mode=cfg.model.input_mode,
                                                     is_test=True)
    

    global_rank = 0
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)


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
        log_writer = wandb.init(project='Sensor-Classification',
                                    config=OmegaConf.to_container(cfg, resolve=True),
                                    dir=cfg.log_dir,
                                    group=f'{cfg.remark}_{task_name}_{unique_id}',
                                    name=f"fold_{0}", # make the wandb groupping happy
                                    reinit=True)
    else:
        log_writer = None
        
    print("Number of Samples:", len(dataset_train))
    
    cfg.batch_size = min(int(cfg.batch_size), len(dataset_train) // misc.get_world_size())
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        collate_fn = collate_fn,
        prefetch_factor=cfg.prefetch_factor,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False,
        collate_fn = collate_fn,
        prefetch_factor=cfg.prefetch_factor,
    )
        

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False,
        collate_fn = test_collate_fn,
        prefetch_factor=cfg.prefetch_factor,
    )
    
    
    if cfg.model.name == 'slip':
        model = instantiate(cfg.model,tokenizer=tokenizer)

    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.llm_model_name, 
                                                     trust_remote_code=True,
                                                      device_map='cpu', 
                                                      torch_dtype='float16',_attn_implementation='flash_attention_2')

        # hardcoded the lora
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        model = get_peft_model(model, lora_config)
    
    if cfg.evaluate:
        try:
            print('loading best model for test evaluation ...')
            model_ckpt = torch.load(cfg.evaluate, map_location='cpu',weights_only=False)
            msg = model.load_state_dict(model_ckpt['model'])
            print(msg)
            
        except:
            print('loading best model for evaluation failed, using the current model weights ...')

        model.to(device)
        model_without_ddp = model
        model.eval()
        
        if task_name == 'tsqa':
            # using MCQ pipeline
            # unlike ecg and activity, each label is few word that easy to normalize and parse. each choice in tsq is a short phrase that hard to parse, and tuning the LM to only output a letter is not stable, thus we instead use the popular mcq evaluation here.
            if cfg.model.name == 'slip':
                result_dict = mcq_test(model, data_loader_test, device)
            else:
                result_dict = chatts_mcq_test(model, tokenizer, data_loader_test, device)
        else:
            # cot pipeline.
            if cfg.model.name == 'slip':
                # SLIP test, (just input format is different..)
                result_dict = sft_test(model, data_loader_test, device)
            else:
                result_dict = chatts_test(model, tokenizer, data_loader_test, device)

        # save the dict to json
        result_json_path = os.path.join(cfg.output_dir, 'test_results.json')
        if misc.get_rank() == 0:
            with open(result_json_path, "w") as f:
                json.dump(result_dict, f, indent=4)

        print(f'Test results saved to {result_json_path}')

        # calculate the accuracy
        if task_name == 'har_cot':
            print('har_cot')
            accuracy = calculate_cot_accuracy(result_dict)
        elif task_name == 'sleep_cot':
            print('sleep_cot')
            accuracy = get_sleep_accuracy(result_dict)
        elif task_name == 'ecg_cot':
            print('get_ecg_accuracy')
            accuracy = get_ecg_accuracy(result_dict)
        elif task_name == 'tsqa':
            print('tsqa')
            accuracy = get_tsqa_accuracy(result_dict)
        if log_writer is not None:
            log_writer.log({'kfold-mean': accuracy,
                            'kfold-std': 0})
        wandb.finish()

        exit()


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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=True)
        # model.module = torch.compile(model.module)
        model_without_ddp = model.module
    
    # Define Optimizer: text decoder has lower lr.
    
    
    if cfg.model.name != 'slip':
        # LLM Branch
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=cfg.betas,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        cfg.is_sft=True
        
    elif len(model_without_ddp.get_lora_parameters())==0:
        # slip branch
        optimizer = torch.optim.AdamW(
            model.parameters(),
            betas=cfg.betas,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            )

    else:
        lora_params = [p for p in model_without_ddp.get_lora_parameters() if p.requires_grad]
        lora_ids = {id(p) for p in lora_params}
        rest_params = [p for p in model_without_ddp.parameters()
                    if p.requires_grad and id(p) not in lora_ids]
        
        lora_lr = cfg.lora_lr if cfg.lora_lr is not None else cfg.lr * 0.1
        param_groups = [{"params": rest_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
                        {"params": lora_params, "lr": lora_lr, "weight_decay": cfg.weight_decay}]

        optimizer =  torch.optim.AdamW(
            param_groups,
            # model.parameters(),
            betas=cfg.betas,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            )
    


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
    
    min_loss = 1e10
    for epoch in range(cfg.start_epoch, cfg.epochs):
        try:
            dataset_train.resample_epoch()
        except:
            if cfg.distributed:
                data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=cfg.max_norm,
            log_writer=log_writer,
            args=cfg)

        
        val_stats = evaluate_sft(model, data_loader_val, device, log_writer=log_writer)
        val_loss = val_stats['loss']

        if val_loss < min_loss:
            min_loss = val_loss

            if cfg.output_dir and misc.is_main_process():
                misc.save_model(
                        args=cfg, model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler, epoch="best"
                    )
                
            print(f'Min loss so far: {min_loss:.4f}')

        if cfg.output_dir and (epoch % cfg.save_every_epoch == 0 or epoch + 1 == cfg.epochs):
            misc.save_model(
                args=cfg, model=model, 
                model_without_ddp=model_without_ddp, 
                optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        

    # --- wait for the save---
    if dist.is_initialized():
        print(f"Rank {misc.get_rank()} waiting at barrier...")
        dist.barrier() 
    # -------------------------
    print('loading best model for test evaluation ...')
    
    if cfg.model.name == 'slip':
        # SLIP model
        model = instantiate(cfg.model,tokenizer=tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.llm_model_name, 
                                                     trust_remote_code=True, 
                                                     device_map='cpu', 
                                                     torch_dtype='float16',
                                                     _attn_implementation='flash_attention_2')
        # hardcoded the lora
        model = get_peft_model(model, lora_config)


    best_model_path = os.path.join(cfg.output_dir, 'checkpoint-best.pth')
    model_ckpt = torch.load(best_model_path, map_location='cpu',weights_only=False)
    msg = model.load_state_dict(model_ckpt['model'])
    print(msg)
    model.to(device)
    model_without_ddp = model
    model.eval()
    
    if task_name == 'tsqa':
        result_dict = mcq_test(model, data_loader_test, device)
    else:
        if cfg.model.name == 'slip':
            # SLIP test, (just input format is different..)
            result_dict = sft_test(model, data_loader_test, device)
        else:
            # LLM Branch, remove the answer completely from the input instead of mask
            result_dict = chatts_test(model, tokenizer, data_loader_test, device)

        
    # save the dict to json
    # calculate the accuracy
    if misc.is_main_process():
        result_json_path = os.path.join(cfg.output_dir, 'test_results.json')
        
        os.makedirs(cfg.output_dir, exist_ok=True)
        
        with open(result_json_path, 'w') as f:
            json.dump(result_dict, f, indent=4)
        
        print(f'Test results saved to {result_json_path}')
        if task_name == 'har_cot':
            print('har_cot')
            accuracy = calculate_cot_accuracy(result_dict)
        elif task_name == 'sleep_cot':
            print('sleep_cot')
            accuracy = get_sleep_accuracy(result_dict)
        elif task_name == 'ecg_cot':
            print('get_ecg_accuracy')
            accuracy = get_ecg_accuracy(result_dict)
        elif task_name == 'tsqa':
            print('tsqa')
            accuracy = get_tsqa_accuracy(result_dict)
        

        # print(f'Global accuracy:, {max_acc:.4f}')
        if log_writer is not None:
            log_writer.log({'kfold-mean': accuracy,
                            'kfold-std': 0})
    
    wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


import torch
import torch.distributed as dist

def sft_test(model, data_loader, device):
    model.eval()
    core = model.module if hasattr(model, "module") else model
    tokenizer = core.tokenizer

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    local_results = []
    with torch.no_grad():
        for data_dict in metric_logger.log_every(data_loader, 50, header):
            sensor = data_dict["sensor"]
            text = data_dict["text"]
            

            # masked out answer to prevent cheating.
            # labels = text["labels"]
            # is_answer = (labels != -100)
            # text["input_ids"][is_answer] = tokenizer.pad_token_id
            # text["attention_mask"][is_answer] = 0

            # #decode the prompt to see what is it
            # # For Debugging ##
            # # prompt_texts = tokenizer.batch_decode(text["input_ids"], skip_special_tokens=True)
            # #print(prompt_texts)
            # #exit()
            # ##

            # Move tensors to device
            sensor = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in sensor.items()}
            text = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in text.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                generated_text = core.generate(text, sensor)

            # ignore_index = -100
            # pad_id = tokenizer.pad_token_id

            # labels = labels.clone()
            # labels[labels == ignore_index] = pad_id

            #Decoding
            text_labels = text["labels"] # list of string
            # text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) 
            text_preds = tokenizer.batch_decode(generated_text, skip_special_tokens=True)   

            for pred, gt in zip(text_preds, text_labels):
                local_results.append({'pred': pred, 'gt': gt})

    # --- Start Multi-GPU Gathering ---
    if dist.is_initialized():
        # Ensure all GPUs have finished inference
        dist.barrier()
        
        # Gather lists from all GPUs. 
        # all_gather_object returns a list where each element is the local_results from a specific rank.
        world_size = dist.get_world_size()
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, local_results)
        
        # Flatten the list of lists into a single list of dicts
        # Only rank 0 necessarily needs to return the full list, 
        # but all_gather_object makes the full list available on all ranks.
        final_results = [item for sublist in gathered_results for item in sublist]
    else:
        final_results = local_results

    return final_results


from tqdm import tqdm
def chatts_test(model, tokenizer, data_loader, device):
    model = torch.compile(model)
    core = model.module if hasattr(model, "module") else model

    header = "Test:"

    local_results = []
    with torch.no_grad():
        for data_dict in tqdm(data_loader, desc=header):
            text = data_dict['text']
            # labels = text['labels']

            # decode the prompt to see what is it
            ## For Debugging ##
            # prompt_texts = tokenizer.batch_decode(text["input_ids"], skip_special_tokens=True)
            # print(prompt_texts)
            # exit()
            ##

            # Move tensors to device
            input_dict = {
                'input_ids': text['input_ids'].to(device),
                'attention_mask': text['attention_mask'].to(device),
                'timeseries': text['timeseries'].to(device) if text['timeseries'] is not None else None,
            }

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                generated_text = core.generate(**input_dict,
                                               max_new_tokens=300, 
                                               num_beams=1, 
                                               do_sample=False,
                                               early_stopping=False)

            # ignore_index = -100
            # pad_id = tokenizer.pad_token_id
            # labels = labels.clone()
            # labels[labels == ignore_index] = pad_id

            # Decoding
            # text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) 
            text_labels = text["labels"]
            text_preds = tokenizer.batch_decode(generated_text, skip_special_tokens=True)   

            for pred, gt in zip(text_preds, text_labels):
                local_results.append({'pred': pred, 'gt': gt})
    # --- Start Multi-GPU Gathering ---
    if dist.is_initialized():
        # Ensure all GPUs have finished inference
        dist.barrier()
        
        # Gather lists from all GPUs. 
        # all_gather_object returns a list where each element is the local_results from a specific rank.
        world_size = dist.get_world_size()
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, local_results)
        
        # Flatten the list of lists into a single list of dicts
        # Only rank 0 necessarily needs to return the full list, 
        # but all_gather_object makes the full list available on all ranks.
        final_results = [item for sublist in gathered_results for item in sublist]
    else:
        final_results = local_results

    return final_results


def extract_activity_label(prediction: str) -> str:
    """Extract the activity label from the model prediction."""
    # Look for "Answer: " pattern
    if "Answer: <class label>" in prediction:
        # Extract everything after "Answer: "
        answer_part = prediction.split("Answer: <class label>")[-1].strip()
        # Take the first word as the activity label
        try:
            label = answer_part.split()[1].strip().lower()
            return label
        except IndexError:
            pass
    else:
        # label branch

        # If no "Answer:" pattern, try to extract the last word as the activity
        words = prediction.strip().split()
        if words:
            return words[-1].lower().strip(string.punctuation)
        else:
            return "unknown"


import re
def extract_cot_activity_label(prediction: str) -> str:
    if not prediction:
        return "unknown"

    text = prediction.lower().strip()

    # 1. Prefer the LAST occurrence of "answer"
    # This handles repeated hallucinated answers earlier in the text
    answer_matches = list(
        re.finditer(r"answer\s*[:.\-]?\s*([a-z_ ]+)", text)
    )

    if answer_matches:
        raw = answer_matches[-1].group(1)
    else:
        # 2. Fallback: take the last word-like token in the text
        raw = text.split()[-1]

    # 3. Clean numeric suffixes and punctuation
    raw = re.sub(r"[^a-z_ ]", "", raw)

    # 4. Normalize spacing variants
    raw = raw.strip().replace(" ", "_")

    # 5. Handle common corruptions like walking upression
    # Keep only the leading alphabetic chunk
    raw = re.match(r"[a-z_]+", raw).group(0) if re.match(r"[a-z_]+", raw) else raw

    return raw if raw else "unknown"


def calculate_cot_accuracy(results):
    num_correct = 0
    num_total = len(results)

    for row in results:
        # Standardize Ground Truth
        # gt_label = extract_activity_label(row['gt'])
        gt_label = extract_cot_activity_label(row['gt'])
        pred = extract_cot_activity_label(row['pred'])

        if pred == gt_label:
            num_correct += 1
        
        else:
            pass

    accuracy = (num_correct / num_total * 100) if num_total > 0 else 0
    
    print("-" * 35)
    print(f"Rigorous Anchor-Based Results")
    print("-" * 35)
    print(f"Total Samples:  {num_total}")
    print(f"Correct:        {num_correct}")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print("-" * 35)

    return accuracy

def calculate_accuracy(pred_dict):
    '''
    Docstring for calculate_accuracy
    
    pred: generated prediction strings
    gt: ground truth strings
    '''

    num_correct = 0
    num_total = len(pred_dict)

    for row in pred_dict:
        pred = extract_activity_label(row['pred'])
        gt = extract_activity_label(row['gt'])
        
        if pred == gt:
            num_correct += 1
        else:
            # print(f'Pred: {pred} | GT: {gt}')
            pass
    
    accuracy = (num_correct / num_total) if num_total > 0 else 0.0
    accuracy = round(accuracy*100, 2)

    print("-" * 35)
    print(f"Total Samples:  {num_total}")
    print(f"Correct:        {num_correct}")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print("-" * 35)

    return accuracy


## SLEEP ##
def _canonicalize_label(text):
    """Return canonical label with stage 4 merged into stage 3.

    - Case-insensitive
    - Trims whitespace and trailing period
    - Merges "non-rem stage 4" -> "Non-REM stage 3"
    - Returns (canonical_label_str, is_supported_bool)
    """
    FALLBACK_LABELS = ['Wake', 'Non-REM stage 1', 'Non-REM stage 2', 'Non-REM stage 3', 'REM sleep', 'Movement']
    SUPPORTED_LABELS = []

    if text is None:
        return "", False

    cleaned = str(text).strip()
    # Remove any end-of-text tokens and trailing period
    cleaned = re.sub(r"<\|.*?\|>|<eos>$", "", cleaned).strip()
    cleaned = re.sub(r"\.$", "", cleaned).strip()

    lowered = cleaned.lower()

    # Normalize common variants and merge stage 4 into stage 3
    if "non-rem" in lowered or "nrem" in lowered:
        # unify spacing/hyphenation
        lowered = lowered.replace("nrem", "non-rem")
        lowered = lowered.replace("non rem", "non-rem")

    # Map stage 4 -> stage 3
    if "non-rem" in lowered and "stage 4" in lowered:
        canonical = "Non-REM stage 3"
    elif "non-rem" in lowered and "stage 3" in lowered:
        canonical = "Non-REM stage 3"
    elif "non-rem" in lowered and "stage 2" in lowered:
        canonical = "Non-REM stage 2"
    elif "non-rem" in lowered and "stage 1" in lowered:
        canonical = "Non-REM stage 1"
    elif "rem" in lowered and "sleep" in lowered:
        canonical = "REM sleep"
    elif lowered in {"wake", "awake"}:
        canonical = "Wake"
    elif "movement" in lowered or lowered == "mov" or lowered == "mt":
        canonical = "Movement"
    else:
        # If it exactly matches a supported label ignoring case, keep it
        # Use fallback labels if supported labels haven't been determined yet
        label_set = SUPPORTED_LABELS if SUPPORTED_LABELS else FALLBACK_LABELS
        maybe = next((lab for lab in label_set if lab.lower() == lowered), "")
        canonical = maybe if maybe else cleaned

    # Use fallback labels if supported labels haven't been determined yet
    label_set = SUPPORTED_LABELS if SUPPORTED_LABELS else FALLBACK_LABELS
    is_supported = canonical in label_set
    return canonical if canonical else cleaned, is_supported

def get_sleep_accuracy(results):
    def extract_answer(text):
        """Extract the final answer from text"""
        if "Answer: " not in text:
            return text

        answer = text.split("Answer: ")[-1].strip()

        # Remove any end-of-text tokens
        answer = re.sub(r"<\|.*?\|>|<eos>$", "", answer).strip()

        # Cut at the first dot followed by anything (single or multiple dots)
        answer = re.split(r"\.", answer, maxsplit=1)[0].strip()

        return answer

    num_correct = 0
    num_total = len(results)
    for row in results:
        gt_label = row['gt']
        pred = row['pred']
        # Extract answers from both fields
        model_prediction_raw = extract_answer(pred)
        if model_prediction_raw.startswith("Wake"):
            model_prediction_raw = "Wake"
        
        ground_truth_raw = extract_answer(gt_label)

        # Canonicalize labels and merge stage 4 -> stage 3
        pred_canon, pred_supported = _canonicalize_label(model_prediction_raw)
        gt_canon, gt_supported = _canonicalize_label(ground_truth_raw)

        # Calculate accuracy (exact match)
        
        accuracy = (pred_canon == gt_canon) and gt_supported
        if accuracy:
            num_correct += 1
            # print(f"GT: {gt_canon} | Pred: {pred_canon}")
        else:
            pass
            print(f"GT: {gt_canon} | Pred: {pred_canon}")

    final_accuracy = (num_correct / num_total ) * 100

    return final_accuracy

# ECG-QA
def get_ecg_accuracy(results):
    def extract_answer(text):
        """Extract the final answer from text"""
        if "Answer: " not in text:
            return text

        answer = text.split("Answer: ")[-1].strip()

        # Remove any end-of-text tokens
        answer = re.sub(r"<\|.*?\|>|<eos>$", "", answer).strip()

        # Cut at the first dot followed by anything (single or multiple dots)
        answer = re.split(r"\.", answer, maxsplit=1)[0].strip()

        # special handling for yes/no answers
        if answer.lower().startswith("yes"):
            answer = "yes"
        elif answer.lower().startswith("no"):
            answer = "no"
        
        # handle pr interval
        elif answer.lower().startswith("pr interval"):
            answer = "pr interval"
        elif answer.lower().startswith("p duration"):
            answer = "p duration"

        return answer

    num_correct = 0
    num_total = len(results)
    for row in results:
        gt_label = row['gt']
        pred = row['pred']
        # Extract answers from both fields
        model_prediction_raw = extract_answer(pred)
        ground_truth_raw = extract_answer(gt_label)

        
        if model_prediction_raw == ground_truth_raw:
            num_correct += 1
            # print(f"GT: {gt_canon} | Pred: {pred_canon}")
        else:
            pass
            print(f"GT: {model_prediction_raw} | Pred: {ground_truth_raw}")

    final_accuracy = (num_correct / num_total ) * 100

    return final_accuracy


## TSQA ##
def extract_choice(pred: str) -> str:
    if not pred:
        return ""

    text = pred.strip().lower()

    # find anchor
    if "answer:" not in text:
        return ""

    after = text.split("answer:", 1)[1]
    after = after.replace(" ", "").strip()

    # strongest match first
    m = re.match(r"\([a-z]\)", after)
    if m:
        return m.group(0)

    m = re.match(r"[a-z]\)", after)
    if m:
        return m.group(0)

    m = re.match(r"[a-z]", after)
    if m:
        return m.group(0)

    return ""


def get_tsqa_accuracy(results):
    num_correct = 0
    num_total = len(results)

    for row in results:
        pred = row['pred'].strip().lower()

        gt = row["gt"].replace("<|im_end|>", "")
        gt = gt.strip().lower()

        if pred == gt:
            num_correct += 1
        else:
            print(f'Pred: {pred} | GT: {gt}')

    accuracy = (num_correct / num_total) if num_total > 0 else 0.0
    accuracy = round(accuracy * 100, 2)

    print("-" * 35)
    print(f"Total Samples:  {num_total}")
    print(f"Correct:        {num_correct}")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print("-" * 35)

    return accuracy


# mcq testing pipeline
def mcq_test(model, data_loader, device, choices=['a', 'b', 'c']):
    model.eval()
    core = model.module if hasattr(model, "module") else model
    tokenizer = core.tokenizer

    # 1. Prepare choice token IDs
    # Note: Some tokenizers prepend a space. add_special_tokens=False is usually best.
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]
    choice_ids_t = torch.tensor(choice_ids).to(device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "MCQ Eval:"

    local_results = []
    
    with torch.no_grad():
        for data_dict in metric_logger.log_every(data_loader, 50, header):
            sensor = data_dict["sensor"]
            text = data_dict["text"]
            labels = text["labels"]

            # 2. Mask out answer to prevent cheating (just like your SFT pipeline)
            # We want the model to predict the token at the position where the label was.
            is_answer = (labels != -100)
            
            # Find the index of the first answer token to get the correct logit position
            # This is typically the token right after "Answer:"
            answer_indices = is_answer.long().argmax(dim=-1) 
            
            # Mask the answer tokens in input_ids
            text["input_ids"][is_answer] = tokenizer.pad_token_id
            text["attention_mask"][is_answer] = 0

            # Move tensors to device
            sensor = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in sensor.items()}
            text = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in text.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # 3. Multimodal Conditioning (as per your SFT logic)
                sensor_hidden, _ = core.embed_sensor(
                    sensors=sensor['input_ids'],
                    sensor_attn_mask=sensor['attention_mask'], 
                    time_index=sensor['time_index']
                )
                core.multimodalModel.condition_image(sensor_hidden)

                # 4. Forward Pass (Get Logits instead of Generate)
                outputs = core.multimodalModel.model(
                    input_ids=text['input_ids'],
                    attention_mask=text['attention_mask'],
                    return_dict=True
                )
                
                # 5. Extract logits for the choice tokens
                # We look at the logit at the position BEFORE the label was masked, 
                # or the very last non-padded token.
                # If your prompt ends with "Answer:", we want the logits for the token immediately following it.
                
                # Option A: Use the exact position where the label started
                batch_range = torch.arange(text['input_ids'].size(0), device=device)
                logit_positions = answer_indices - 1 # Position of the token that predicts the answer
                
                all_logits = outputs.logits[batch_range, logit_positions, :]
                candidate_logits = all_logits[:, choice_ids_t] # (BS, num_choices)
                
                # 6. Get predicted choice
                pred_indices = torch.argmax(candidate_logits, dim=-1)
                text_preds = [choices[i] for i in pred_indices.cpu().numpy()]

            # 7. Decode Ground Truth
            labels_copy = labels.clone()
            labels_copy[labels_copy == -100] = tokenizer.pad_token_id
            text_labels = tokenizer.batch_decode(labels_copy, skip_special_tokens=True)
            
            # Clean up labels (in case they contain more than just the letter)
            text_labels = [gt.strip() for gt in text_labels]

            for pred, gt in zip(text_preds, text_labels):
                local_results.append({'pred': pred, 'gt': gt})

    # --- Multi-GPU Gathering (Same as your reference) ---
    if dist.is_initialized():
        dist.barrier()
        world_size = dist.get_world_size()
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, local_results)
        final_results = [item for sublist in gathered_results for item in sublist]
    else:
        final_results = local_results

    return final_results


def chatts_mcq_test(model, tokenizer, data_loader, device, 
                    choices=['a', 'b', 'c']):
    model.eval()
    core = model.module if hasattr(model, "module") else model

    # 1. Prepare choice token IDs
    # Note: Some tokenizers prepend a space. add_special_tokens=False is usually best.
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]
    choice_ids_t = torch.tensor(choice_ids).to(device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "MCQ Eval:"

    local_results = []
    
    with torch.no_grad():
        for data_dict in metric_logger.log_every(data_loader, 50, header):
            text = data_dict["text"]
            input_dict = {
                'input_ids': text['input_ids'].to(device),
                'attention_mask': text['attention_mask'].to(device),
                'timeseries': text['timeseries'].to(device) if text['timeseries'] is not None else None,
            }

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output_dict = core(**input_dict,return_dict=True)
                all_logits = output_dict.logits  # (BS, seq_len, vocab_size)
                all_logits = all_logits[:, -1, :]  # (BS, vocab_size) - take the last token's logits
                candidate_logits = all_logits[:, choice_ids_t] # (BS, num_choices)
                
                # 6. Get predicted choice
                pred_indices = torch.argmax(candidate_logits, dim=-1)
                text_preds = [choices[i] for i in pred_indices.cpu().numpy()]

            
            text_labels = text["labels"]

            for pred, gt in zip(text_preds, text_labels):
                local_results.append({'pred': pred, 'gt': gt})

    # --- Multi-GPU Gathering (Same as your reference) ---
    if dist.is_initialized():
        dist.barrier()
        world_size = dist.get_world_size()
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, local_results)
        final_results = [item for sublist in gathered_results for item in sublist]
    else:
        final_results = local_results

    return final_results





if __name__ == '__main__':
    main()
