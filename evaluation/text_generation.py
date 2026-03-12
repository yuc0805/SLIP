import re
from util.dataset import LlmPromptingDataset, TextGenerationCollator
import os
import time
import datetime
from pathlib import Path
import uuid
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import util.misc as misc
from transformers import AutoTokenizer, AutoModelForCausalLM

WANDB_API_KEY = os.getenv("wandb_key")
wandb.login(key=WANDB_API_KEY)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(version_base=None, config_path="../config", config_name="zs_caption.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False

    dataset_name = cfg.dataset.name
    if dataset_name == 'pm':
        if cfg.dataset.fold is not None:
            fold_dict = cfg.dataset.fold

        task_name = cfg.dataset.task_name
        task_config = cfg.dataset.task_config[task_name]
        y_range = task_config.get('y_range', None)
    else:
        raise NotImplementedError('The specified dataset is not implemented.')

    # setup run name''' build logging information'''
    if cfg.remark is None:
        remark = f"zeroshot_{dataset_name}_{task_name}"
    else:
        remark = f"zeroshot_{cfg.remark}_{dataset_name}_{task_name}"

    print('model remark is', remark)

    ### ### ###
    # prepare logging.
    unique_id = uuid.uuid4().hex[:8] 
    root_output_dir = cfg.output_dir
    root_log_dir = cfg.log_dir
    # Begin k-fold.
    global_acc = []
    for fold_idx, (fold_key, split) in enumerate(fold_dict.items()):
        print('running fold', fold_idx, fold_key)
        train_subjects = split['train']
        val_subjects = split['val']

        label_map = task_config.get('label_map', None)
        if label_map is not None:
            label_map = {int(k): int(v) for k, v in label_map.items()}


        # dataset_val = PmTextGenerationDataset(
        #     data_dir=cfg.dataset.data_dir,
        #     subject_list=val_subjects,
        #     task_name= task_name,
        #     feature_len=cfg.dataset.input_size[1],
        #     transform=None,
        #     label_map=label_map,
        #     prompt_template_path=cfg.dataset.prompt_template_path,
        #     subject_demo_path=cfg.dataset.subject_demo_path,
        #     tokenize_method=cfg.dataset.tokenize_method
        #     )

        dataset_val = LlmPromptingDataset(
            data_dir=cfg.dataset.data_dir,
            )
        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name, use_fast=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            # ensure pad token is defined
            tokenizer.pad_token = tokenizer.eos_token
            print('pad token not found, set pad token to eos token')


        collate_fn = TextGenerationCollator(tokenizer, max_len=cfg.max_llm_len)
        print('Using dataset',dataset_name)
 
        print("Number of Testing Samples:", len(dataset_val))

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        # prepare log ----------------------------------------------
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        cfg.log_dir = os.path.join(root_log_dir, remark, timestamp)
        cfg.output_dir = os.path.join(root_output_dir, remark, timestamp)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        if cfg.log_dir is not None:
            log_writer = wandb.init(project='Sensor-Classification',
                                    config=OmegaConf.to_container(cfg, resolve=True),
                                    dir=cfg.log_dir,
                                    group=f'{remark}_{task_name}_{unique_id}',
                                    name=f"fold_{fold_idx}",
                                    reinit=True)
        else:
            log_writer = None

       
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=8,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            drop_last=False,
            collate_fn=collate_fn,)

        model = AutoModelForCausalLM.from_pretrained(cfg.llm_model_name)
        model.to(device)
        model.eval()
        
        pred = []
        gt = []

        # evaluate -----------------
        for batch in data_loader_val:
            prompt = batch['prompt']
            targets = batch['targets']
            gt.extend(targets.numpy().tolist())

            for key, value in prompt.items():
                prompt[key] = value.to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **prompt,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=cfg.do_sample,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,    
                    temperature=cfg.temperature,
                    num_beams=cfg.num_beams,
                    early_stopping=cfg.early_stopping,
                    repetition_penalty= cfg.repetition_penalty,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_texts = tokenizer.batch_decode(output_ids.detach().cpu(), skip_special_tokens=True)
            decoded_prompts = tokenizer.batch_decode(prompt["input_ids"].detach().cpu(), skip_special_tokens=True)

            clamped_count = 0 
            for sample_idx, (prompt_text, generated_text) in enumerate(zip(decoded_prompts, generated_texts)):
                # remove prompt from full generated text
                if generated_text.startswith(prompt_text):
                    answer_text = generated_text[len(prompt_text):].strip()
                else:
                    answer_text = generated_text.strip()

                # extract numeric score
                score = re.search(r"[-+]?\d+(\.\d+)?", answer_text)
                if score:
                    value = float(score.group())
                    original_value = value

                    # clamp if necessary
                    if y_range is not None:
                        clamped_value = max(y_range[0], min(y_range[1], value))
                        if clamped_value != value:
                            clamped_count += 1
                            print(f"[Clamp] Original: {original_value:.3f} → Clamped: {clamped_value:.3f} | Text: '{generated_text}'")
                        value = clamped_value
                        

                    pred.append(value)
                else:
                    print(f"[Warning] No valid number found in generated text: '{generated_text}'")
                    clamped_count += 1
                    pred.append(0.0)
                    

            # print summary
            if y_range is not None:
                print(f"\nTotal clamped scores: {clamped_count}/{len(decoded_prompts)}")

                


        # compute metrics
        print(generated_texts[0]) # print an example
        if task_config.head_type == 'classification':
            pred = np.array(pred)
            gt = np.array(gt)
            correct = (pred == gt).sum()
            total = len(gt)
            acc = (correct / total)*100
            global_acc.append(acc)
            results = {'accuracy': acc}

            print(f"\n[Classification Summary]")
            unique, counts = np.unique(pred, return_counts=True)
            dist = {int(k): int(v) for k, v in zip(unique, counts)}
            print(f"Predicted class distribution: {dist}")
            print(f"Accuracy: {acc:.2f}")

        else:
            pred = np.array(pred)
            gt = np.array(gt)
            mae = np.abs(gt - pred).mean()
            mse = ((gt - pred)**2).mean()
            results = {'mae': mae, 'mse': mse}
            global_acc.append(mae)

            # --- smart summary print ---
            print(f"\n[Regression Summary]")
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")
            print(f"Pred mean: {pred.mean():.4f}, std: {pred.std():.4f}, min: {pred.min():.2f}, max: {pred.max():.2f}")

            # --- mode prediction check ---
            # If many values are identical (or almost identical), it's likely mode prediction
            rounded_pred = np.round(pred, 2)  # round to 2 decimals to detect near-identical predictions
            unique, counts = np.unique(rounded_pred, return_counts=True)
            most_common_value = unique[np.argmax(counts)]
            most_common_ratio = counts.max() / len(pred)

            print(f"Most common predicted value: {most_common_value:.2f} (appears {most_common_ratio*100:.1f}% of the time)")

            if most_common_ratio > 0.5:
                print("⚠️  Warning: Model may be doing mode prediction (low output diversity).")

            # Optional: show top 10 most frequent predictions
            sorted_counts = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]
            print("Top predicted values:")
            for val, cnt in sorted_counts:
                print(f"  {val:.2f}: {cnt} samples")
                

        print(f"Task {task_name} Fold {fold_idx} results:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

        # log results
        if log_writer is not None:
            log_writer.log({f'evaluation_{k}': v for k, v in results.items()})

    print(f'Global accuracy:, {np.mean(global_acc):.4f} +- {np.std(global_acc):.4f}')
    log_writer.log({'kfold-mean': np.mean(global_acc), 'kfold-std': np.std(global_acc)})
    wandb.finish()

    return np.mean(global_acc)


if __name__ == '__main__':
    main()

    


        
        
