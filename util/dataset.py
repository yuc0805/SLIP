import json
import os
from pathlib import Path
import h5py
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
from collections import Counter, defaultdict
from typing import Tuple, Dict, Any, List
from string import Template
from data_generator.data_to_attribute import get_attribute_data
from util.serialize import serialize_arr,SerializerSettings
from einops import rearrange
import pandas as pd
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Union, Callable

def build_time_index(attention_mask):
    '''
    attention_mask: nvar, L
    
    '''
    L = attention_mask.shape[-1]
    positions = torch.arange(1, L+1, device=attention_mask.device).float()
    positions = positions.unsqueeze(0).expand_as(attention_mask)      # nvar, L
    time_index = positions / (L+1)

    return time_index

def prepare_patches(sensor,patch_size,return_patches=True):
    '''
    sensor:nvar,L
    '''
    sensor = torch.as_tensor(sensor).float()
    mask = torch.ones_like(sensor, dtype=torch.bool)
    sensor = torch.nan_to_num(sensor, nan=0.0)
    mask = mask & (~torch.isnan(sensor))
    time_index = build_time_index(mask)

    L = sensor.shape[-1]
    remainder = L % patch_size
    if remainder != 0:
        pad_len = patch_size - remainder
        # pad zeros to the left
        sensor = F.pad(sensor, (pad_len, 0), mode='constant', value=0.0) # C, L
        mask = F.pad(mask, (pad_len, 0), mode='constant', value=False)
        time_index = F.pad(time_index, (pad_len, 0), mode='constant', value=0.0)

    if return_patches:
        sensor = rearrange(sensor, 'c (n p) -> c n p', p=patch_size)
        mask = rearrange(mask, 'c (n p) -> c n p', p=patch_size)
        time_index = rearrange(time_index, 'c (n p) -> c n p', p=patch_size)

    return sensor, mask, time_index
    


def EvalCollator(batch,return_patch=True):
    # Filter out invalid entries
    filtered = [item for item in batch if item["targets"] is not None and not torch.all(item["targets"] == -1)]
    targets = torch.stack([item["targets"] for item in filtered], dim=0)

    if len(filtered) == 0:
        raise ValueError("All targets are None or -1 in the batch.")

    sensors = []
    masks = []
    time_indices = []
    for item in filtered:
        samples = item["samples"]  
        mask = item["mask"]
        time_index = item["time_index"]

        sensors.append(samples)
        masks.append(mask)
        time_indices.append(time_index)

    if not return_patch:
        sensors = torch.stack(sensors, dim=0)  # BS, nvar, L
        masks    = torch.stack(masks, dim=0)
        time_indices = torch.stack(time_indices, dim=0)

    return {
        'samples': sensors,
        'mask': masks,
        'time_index': time_indices,
        "targets": targets
    }


def ChronosCollator(batch):
    # Filter out invalid entries
    filtered = [item for item in batch if item["targets"] is not None and not torch.all(item["targets"] == -1)]
    targets = torch.stack([item["targets"] for item in filtered], dim=0)

    if len(filtered) == 0:
        raise ValueError("All targets are None or -1 in the batch.")

    sensors = []
    masks = []
    time_indices = []
    for item in filtered:
        samples = item["samples"] 
        samples = rearrange(samples, 'c n p -> c (n p)')  # flatten patches 
        
        mask = item["mask"]
        time_index = item["time_index"]

        sensors.append(samples)
        masks.append(mask)
        time_indices.append(time_index)

    sensors = torch.stack(sensors, dim=0)  # BS, C, L

    

    return {
        'samples': sensors,
        'mask': masks,
        'time_index': time_indices,
        "targets": targets
    }


### LLM-Zeroshot Prompting ####
# from util.prompt_formatter import render_prompt_from_template
import matplotlib.pyplot as plt
import io
from PIL import Image

class MCQDataset(Dataset):
    def __init__(self, data_dir,is_img=False):
        self.data_dir = data_dir

        with open(f"{data_dir}/split_indices.json", "r") as f:
            split_dict = json.load(f)
        self.indices = split_dict['test']
            
        self.data = np.load(f"{data_dir}/X.npy", mmap_mode="r")
        self.label = np.load(f"{data_dir}/text_y.npy") 
    
        self.candidates = sorted(list(np.unique(self.label).astype(str)))
        self.label_map = {c: i for i, c in enumerate(self.candidates)}

        with open(f"{data_dir}/mqa_prompt.json", "r", encoding="utf-8") as f:
            prompt_template = json.load(f)
        
        self.question_header = prompt_template['question_header']
        self.prompt_template = prompt_template['context']
        # find the list of nvar name
        # find the word before {data[0]}, ignore the space and :
        pat = re.compile(r"\n\s*([^\n:]+?)\s*:\s*\{data\[(\d+)\]\}")
        pairs = [(int(idx), name.strip()) for (name, idx) in pat.findall(self.prompt_template)]
        if not pairs:
            raise ValueError("No lines like 'name: {data[k]}' found in prompt template.")

        pairs.sort(key=lambda x: x[0])
        self.nvar_names = [name for _, name in pairs]

        self.is_img = is_img

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sensor = self.data[real_idx].astype(np.float32) 
        label_idx = self.label_map[str(self.label[real_idx])]
        choice_list = "\n".join([f"{chr(65+i)}. {cand}" for i, cand in enumerate(self.candidates)])

        # THE FIX: Define a strict System Message
        system_message = {
            "role": "system", 
            "content": "You are a precise sensor data classifier. You must respond with only a single letter (A, B, C, D...) and nothing else. Never explain your reasoning."
        }

        if self.is_img:
            # --- IMAGE MODE ---
            prompt_header = self.prompt_template.split("\n\n")[0]
            image = self.plot_sensor_data(sensor)
            user_content = [
                {"type": "image"},
                {"type": "text", "text": f"{prompt_header}\n\nOptions:\n{choice_list}\n\nTask: Output the letter of the correct activity.\nAnswer:"}
            ]
            messages = [system_message, {"role": "user", "content": user_content}]
            return {'prompt': messages, 'image': image, 'label': label_idx, 'is_img': True}

        else:
            # --- TEXT MODE ---
            # Render the sensor data as text context
            context = render_prompt_from_template(self.prompt_template, sensor)
            user_content = [
                {"type": "text", "text": f"{context.strip()}\n\nOptions:\n{choice_list}\n\nTask: Output the letter of the correct activity.\nAnswer:"}
            ]
            messages = [system_message, {"role": "user", "content": user_content}]
            # Return None for image so the collator knows there is no visual data
            return {'prompt': messages, 'image': None, 'label': label_idx, 'is_img': False}

    def plot_sensor_data(self, sensor_array):
        """Converts (nvar, L) sensor array to a PIL Image."""
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

        for i in range(sensor_array.shape[0]):
            name = self.nvar_names[i] if hasattr(self, "nvar_names") and i < len(self.nvar_names) else f"var{i}"
            ax.plot(sensor_array[i], label=name, linewidth=1.2)

        ax.set_title("Sensor Data")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        # cleaner frame
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # legend outside on the right
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
        )

        # leave room for the outside legend
        fig.tight_layout(rect=(0, 0, 0.82, 1))

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

class MCQCollator:
    def __init__(self, processor, is_img=False):
        self.processor = processor
        self.is_img = is_img

    def __call__(self, features):
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        
        # Apply the chat template to EVERY feature
        # This works for both Text and Multimodal messages
        texts = [
            self.processor.apply_chat_template(f["prompt"], add_generation_prompt=True, tokenize=False) 
            for f in features
        ]

        if self.is_img:
            images = [[f["image"]] for f in features] # List of PIL Images
            # Multimodal processing
            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        else:
            # Text-only processing using the processor's tokenizer
            batch = self.processor(text=texts, return_tensors="pt", padding=True)

        batch["labels"] = labels
        return batch
        
from torch.utils.data import Dataset
import json
class ChatTSDataset(Dataset):
    def __init__(self, data_dir,is_zs=True,split='test'):
        self.data_dir = data_dir

        with open(f"{data_dir}/split_indices.json", "r") as f:
            split_dict = json.load(f)
        self.indices = split_dict[split]
            
        self.data = np.load(f"{data_dir}/X.npy", mmap_mode="r")
        self.label = np.load(f"{data_dir}/text_y.npy") 
        self.candidates = sorted(list(np.unique(self.label).astype(str)))
        self.label_map = {c: i for i, c in enumerate(self.candidates)}
        choice_list = "\n".join([f"{chr(65+i)}. {cand}" for i, cand in enumerate(self.candidates)])
        
        if is_zs:
            with open(f"{data_dir}/mqa_prompt.json", "r", encoding="utf-8") as f:
                prompt_template = json.load(f)
            self.prompt_template = prompt_template['context']
            self.prompt_template = re.sub(r"\{data(?:\[[^\]]+\])*\}", "<ts><ts/>", self.prompt_template)
            self.prompt_template = f"{self.prompt_template}. {choice_list}. You must respond with only a single letter (A, B, C, D...) and nothing else. Never explain your reasoning."
            self.prompt_template = f"""<|im_start|>system
            You are a helpful assistant.<|im_end|><|im_start|>user
            {self.prompt_template}<|im_end|><|im_start|>assistant
            """
            print("Prompt template:", self.prompt_template)
        else:
            # load a sample 
            _, nvar, L = self.data.shape
            self.prompt_template = ""
            for i in range(nvar):
                self.prompt_template += f"I have time series {i}: <ts><ts/>, "
            self.prompt_template = self.prompt_template + "What does this suggest?"
            # format the template
            self.prompt_template =f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.prompt_template}<|im_end|>\n<|im_start|>assistant\n"

            print("Prompt template:", self.prompt_template)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sensor = self.data[real_idx].astype(np.float32) # (nvar, L)
        # make this into list of (L,) array
        sensor = [sensor[i] for i in range(sensor.shape[0])]
        label_idx = self.label_map[str(self.label[real_idx])]

        return{
            'sensor': sensor,
            'label': label_idx,
            'prompt': self.prompt_template
        }

def ChatTScollator(processor, batch):
    # flatten all the time series into one list
    ts_list = []
    formatted_prompts = []
    for item in batch:
        ts_list += item['sensor']
        formatted_prompts.append(item['prompt'])

    inputs = processor(text=formatted_prompts, timeseries=ts_list, padding=True, return_tensors="pt")

    return {'inputs': inputs, 
            'label': torch.tensor([item['label'] for item in batch], dtype=torch.long)}


class SlipMCQDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 split='test',
                 # legacy
                 patch_size = 16, 
                 sensor_aug=True, 
                 return_patch=True,
                 is_normalize=True):
        
        self.data_dir = data_dir
        self.return_patch = return_patch
        self.is_normalize = is_normalize
        self.patch_size = patch_size

        with open(f"{data_dir}/split_indices.json", "r") as f:
            split_dict = json.load(f)
        self.indices = split_dict['test']
        self.data = np.load(f"{data_dir}/X.npy", mmap_mode="r")
        self.label = np.load(f"{data_dir}/text_y.npy") 

        self.candidates = sorted(list(np.unique(self.label).astype(str)))
        self.label_map = {c: i for i, c in enumerate(self.candidates)}

        with open(f"{data_dir}/mqa_prompt.json", "r", encoding="utf-8") as f:
            prompt_template = json.load(f)
        
        self.question_header = prompt_template['question_header']
        self.prompt_template = prompt_template['context']

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):

        real_idx = self.indices[idx]
        sensor = self.data[real_idx].astype(np.float32) 
        sensor,mask,time_index = prepare_patches(sensor, self.patch_size)

        label_idx = self.label_map[str(self.label[real_idx])]
        # choice_list = "\n".join([f"{chr(65+i)}. {cand}" for i, cand in enumerate(self.candidates)])
        choice_list = "\n".join([f"{cand}" for i, cand in enumerate(self.candidates)])

        prompt_header = self.prompt_template.split("\n\n")[0]
        content = f"You are a precise sensor data classifier. {prompt_header}\n\nOptions:\n{choice_list}\nAnswer:"

        return {
            'prompt': content,
            'sensor': sensor, # patches
            'mask': mask,
            'time_index': time_index,
            'labels': label_idx
        }
    
def SlipMCQcollator(tokenizer, batch):

    sensors, sensor_masks, time_indices = [], [], []
    prompts = []
    for item in batch:
        # item['sensors] is tensor shape of (nvar, npatch, patch_size)
        # need to make it a list of (npatch, patch size) array
        sensors.append(list(item["sensor"]))
        sensor_masks.append(list(item["mask"]))
        time_indices.append(list(item["time_index"])) 
        prompts.append(item["prompt"].rstrip())

    inputs = tokenizer(text=prompts,
                       padding=True,
                       truncation=True,
                        max_length=2880, 
                        return_tensors="pt",
                    )
                       

    return {'sensor': 
            {
                'input_ids': sensors, # list of list
                'attention_mask': sensor_masks, # list of list
                'time_index': time_indices # list of list
            },
            'text': inputs,
            'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long)}
    




class SLIPDataset(Dataset):
    def __init__(self, 
                 data_map, 
                 meta_info,
                 patch_size=None, 
                 sensor_aug=None, # Changed default to None to match logic
                 is_normalize=True,
                 text_aug=True,
                 sampling=True,
                 return_patches=True,
                 subset_ratio=1.0,
                 mis_align=False):
        
        # Read metadata
        self.data_map = pd.read_json(data_map, lines=True)
        self.return_patches = return_patches
        self.sensor_aug = sensor_aug # Corrected assignment
        
        # Group indices by source file
        grouped_indices = defaultdict(list)
        file_meta = {}
        
        # We need to preserve the original DataFrame index for every file entry
        # Structure: source_path -> list of (index_in_npy, index_in_dataframe)
        for df_idx, row in self.data_map.iterrows():
            grouped_indices[row["source_path"]].append((row["index"], df_idx))
            file_meta[row['source_path']] = (row['category'], row['dataset'])

        self.data = []
        self.map_indices = [] # storing data_map indices
        
        self.use_flexi_patch = True if patch_size is None else False
        
        self.category_list = []
        self.dataset_list = []

        # Setup Patch sizes
        if self.use_flexi_patch:
            self.patch_size_list = [] 
            self.meta_info = pd.read_csv(meta_info)
            self.patch_lookup = {
                (row['Domain'], row['Dataset']): row['patch_size']
                for _, row in self.meta_info.iterrows()
                if not pd.isna(row['patch_size'])
            }
        else:
            self.patch_size_val = patch_size 
            self.max_len = patch_size * 64 
            self.return_patches = False

        self.text_aug = text_aug
        self.is_normalize = is_normalize
        self.mis_align = mis_align

        # Indices tracking
        self.mts_indices = []
        self.uts_indices = []
        self.active_indices = [] 
        
        self.nvar_list = []
        self.length = []

        # Load Data
        for ds_path, index_pairs in grouped_indices.items():
            # index_pairs is list of (npy_idx, df_idx)
            npy_indices = [x[0] for x in index_pairs]
            df_indices = [x[1] for x in index_pairs]
            
            arr = np.load(ds_path) 
            

            npy_indices = np.array(npy_indices, dtype=int)
            selected = arr[..., npy_indices] 

            # Prepare temporary lists for this file
            current_data_list = []
            current_df_indices = []
            current_nvars = []
            current_lens = []

            if selected.ndim == 2:  # (L, #selected)
                L, num_sel = selected.shape
                
                # Iterate individually to filter by length
                for i in range(num_sel):
                    # Length check filter
                    if not self.use_flexi_patch and L > self.max_len:
                        continue
                    
                    current_data_list.append(selected[:, i].reshape(1, L))
                    current_df_indices.append(df_indices[i])
                    current_nvars.append(1)
                    current_lens.append(L)
                    
                    # Track global indices for sampling
                    # Note: we append to uts_indices relative to the growing self.data
                    self.uts_indices.append(len(self.data) + len(current_data_list) - 1)

            else:  # (C, L, #selected)
                C, L, num_sel = selected.shape
                
                for i in range(num_sel):
                    if not self.use_flexi_patch and L > self.max_len:
                        continue
                        
                    current_data_list.append(selected[:, :, i])
                    current_df_indices.append(df_indices[i])
                    current_nvars.append(C)
                    current_lens.append(L)
                    self.mts_indices.append(len(self.data) + len(current_data_list) - 1)

            # Extend main storage
            self.data.extend(current_data_list)
            self.map_indices.extend(current_df_indices)
            self.nvar_list.extend(current_nvars)
            self.length.extend(current_lens)

            # Handle Flexi Patch Size

            if self.use_flexi_patch:
                category, dataset = file_meta[ds_path]
                patch = self.patch_lookup.get((category, dataset), 64) # Default to 64 if missing
                self.patch_size_list.extend([patch] * len(current_data_list))
                self.category_list.extend([category] * len(current_data_list))
                self.dataset_list.extend([dataset] * len(current_data_list))

        self.nvar_list = np.array(self.nvar_list)

        if subset_ratio is not None and float(subset_ratio) < 1.0:
            self._fix_subset(
                subset_ratio=float(subset_ratio),
            )

        if self.mis_align:
            print('Warning! MISALIGN!!!!!!!!!!!')
            rng = np.random.default_rng(0)
            used_df_idx = np.array(self.map_indices, dtype=int)

            caps = self.data_map.loc[used_df_idx, "caption"].to_numpy()
            n = len(caps)

            if n > 1:
                perm = rng.permutation(n)

                same = perm == np.arange(n)
                if np.any(same):
                    same_pos = np.where(same)[0]
                    for i in same_pos:
                        j = int(rng.integers(0, n - 1))
                        if j >= i:
                            j += 1
                        tmp = perm[i]
                        perm[i] = perm[j]
                        perm[j] = tmp

                self.data_map.loc[used_df_idx, "caption"] = caps[perm]

        if sampling:
            self.resample_epoch()
        else:
            self.active_indices = np.arange(len(self.data))
            
    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        # 1. Get the physical index in self.data
        dat_idx = self.active_indices[idx]
    
        # 2. Get the ORIGINAL DataFrame index to ensure alignment
        df_idx = self.map_indices[dat_idx]

        caption = self.data_map['caption'].iloc[df_idx] # Access by integer location

        if self.text_aug and isinstance(caption, list) and len(caption) > 0:
            caption = np.random.choice(caption)
        elif isinstance(caption, list):
            caption = caption[0]
                
        # 5. Handle Sensor Data
        sensor = self.data[dat_idx]
        
        # Normalize
        if self.is_normalize:
            # Added 1e-6 to denominator safe guard
            denom = np.mean(np.abs(sensor), axis=1, keepdims=True) + 1e-6
            sensor = sensor / denom

        sensor = torch.from_numpy(sensor).float() 

        if self.use_flexi_patch:
            patch_size = self.patch_size_list[dat_idx]
        else:
            patch_size = self.patch_size_val

        if self.sensor_aug is not None:
            sensor = self.sensor_aug(sensor)

        # Assumes prepare_patches exists in scope
        sensor, mask, time_index = prepare_patches(sensor, patch_size, return_patches=self.return_patches)
            
        return {
            'caption': caption,
            'sensor': sensor,
            'mask': mask,
            'time_index': time_index,
            'patch_size': patch_size,
        }

    def _fix_subset(self, subset_ratio=1.0):
        print(f"Applying subset ratio of {subset_ratio} to UTSDDataset.")
        """
        Subsets the available indices to a specific ratio. 
        The remaining indices are effectively 'held out' and will not 
        be used by the resample_epoch function.
        """
        # Convert to numpy arrays for easier shuffling and slicing
        uts_indices = np.array(self.uts_indices)
        mts_indices = np.array(self.mts_indices)

        # Shuffle the indices to ensure the subset is random
        np.random.shuffle(uts_indices)
        np.random.shuffle(mts_indices)

        # Calculate the number of samples to keep for the active subset
        num_uts = int(len(uts_indices) * subset_ratio)
        num_mts = int(len(mts_indices) * subset_ratio)

        # Update the class attributes with the subset.
        # resample_epoch will now only pull from these lists.
        self.uts_indices = uts_indices[:num_uts].tolist()
        self.mts_indices = mts_indices[:num_mts].tolist()


    def resample_epoch(self, ratio=2):
        num_mts = len(self.mts_indices)
        if num_mts == 0:
            # Fallback if no MTS data
            self.active_indices = np.array(self.uts_indices)
            np.random.shuffle(self.active_indices)
            return

        num_uts_needed = int(num_mts // ratio)
        
        # Safe sampling
        if num_uts_needed > len(self.uts_indices):
            # If we need more UTS than we have, take them all (or replace=True)
            selected_uts = np.array(self.uts_indices) 
        else:
            selected_uts = np.random.choice(self.uts_indices, size=num_uts_needed, replace=False)
            
        self.active_indices = np.concatenate([self.mts_indices, selected_uts])
        np.random.shuffle(self.active_indices)

class SLIPhfDataset(Dataset):
    def __init__(self,
                 hf_dataset,           # already-loaded HF Dataset (the train split)
                 meta_info=None,        # path to meta_info CSV for flexi patch lookup
                 patch_size=None,
                 is_normalize=True,
                 text_aug=True,
                 sampling=True,
                 return_patches=True,
                 subset_ratio=1.0,
                 mis_align=False):

        self.hf = hf_dataset
        self.return_patches = return_patches
        self.text_aug = text_aug
        self.is_normalize = is_normalize
        self.mis_align = mis_align

        self.use_flexi_patch = patch_size is None

        if self.use_flexi_patch:
            self.patch_size_list = []
            meta = pd.read_csv(meta_info)
            self.patch_lookup = {
                (row['Domain'], row['Dataset']): row['patch_size']
                for _, row in meta.iterrows()
                if not pd.isna(row['patch_size'])
            }
        else:
            self.patch_size_val = patch_size
            self.max_len = patch_size * 64
            self.return_patches = False

        # --- bulk column access (much faster than row-by-row iteration) ---
        print(f"Loading {len(hf_dataset)} samples from HuggingFace dataset...")
        all_ts = hf_dataset["time_series"]
        all_cap0 = hf_dataset["caption0"]
        all_cap1 = hf_dataset["caption1"]
        all_cap2 = hf_dataset["caption2"]
        all_cap3 = hf_dataset["caption3"]
        all_category = hf_dataset["category"]
        all_dataset = hf_dataset["dataset"]
        print("Columns loaded, building index...")

        self.data = []
        self.captions = []
        self.category_list = []
        self.dataset_list = []
        self.nvar_list = []
        self.length = []
        self.uts_indices = []
        self.mts_indices = []

        for i in range(len(all_ts)):
            ts = np.array(all_ts[i], dtype=np.float32)
            if ts.ndim == 1:
                ts = ts[np.newaxis, :]

            nvar, L = ts.shape

            if not self.use_flexi_patch and L > self.max_len:
                continue

            idx = len(self.data)
            self.data.append(ts)
            self.captions.append([all_cap0[i], all_cap1[i], all_cap2[i], all_cap3[i]])
            self.category_list.append(all_category[i])
            self.dataset_list.append(all_dataset[i])
            self.nvar_list.append(nvar)
            self.length.append(L)

            if nvar == 1:
                self.uts_indices.append(idx)
            else:
                self.mts_indices.append(idx)

            if self.use_flexi_patch:
                patch = self.patch_lookup.get((all_category[i], all_dataset[i]), 64)
                self.patch_size_list.append(patch)

        self.nvar_list = np.array(self.nvar_list)
        print(f"Dataset ready: {len(self.data)} samples loaded.")

        # mis-align: shuffle captions relative to sensors
        if self.mis_align:
            print("Warning! MISALIGN!!!!!!!!!!!")
            rng = np.random.default_rng(0)
            n = len(self.captions)
            perm = rng.permutation(n)
            same = np.where(perm == np.arange(n))[0]
            for i in same:
                j = int(rng.integers(0, n - 1))
                if j >= i:
                    j += 1
                perm[i], perm[j] = perm[j], perm[i]
            self.captions = [self.captions[perm[i]] for i in range(n)]

        if subset_ratio is not None and float(subset_ratio) < 1.0:
            self._fix_subset(float(subset_ratio))

        if sampling:
            self.resample_epoch()
        else:
            self.active_indices = np.arange(len(self.data))

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        dat_idx = self.active_indices[idx]

        # --- text ---
        caption_list = self.captions[dat_idx]
        # filter out empty strings from missing captions
        valid = [c for c in caption_list if c]
        if self.text_aug and valid:
            caption = np.random.choice(valid)
        else:
            caption = valid[0] if valid else ""

        # --- sensor ---
        sensor = self.data[dat_idx].copy()  # (nvar, length)

        if self.is_normalize:
            denom = np.mean(np.abs(sensor), axis=1, keepdims=True) + 1e-6
            sensor = sensor / denom

        sensor = torch.from_numpy(sensor).float()

        patch_size = (
            self.patch_size_list[dat_idx] if self.use_flexi_patch
            else self.patch_size_val
        )

        sensor, mask, time_index = prepare_patches(
            sensor, patch_size, return_patches=self.return_patches
        )

        return {
            "caption":       caption,
            "sensor":        sensor,
            "mask":          mask,
            "time_index":    time_index,
            "patch_size":    patch_size,
        }

    def _fix_subset(self, subset_ratio):
        print(f"Applying subset ratio of {subset_ratio}.")
        uts = np.array(self.uts_indices)
        mts = np.array(self.mts_indices)
        np.random.shuffle(uts)
        np.random.shuffle(mts)
        self.uts_indices = uts[:int(len(uts) * subset_ratio)].tolist()
        self.mts_indices = mts[:int(len(mts) * subset_ratio)].tolist()

    def resample_epoch(self, ratio=2):
        num_mts = len(self.mts_indices)
        if num_mts == 0:
            self.active_indices = np.array(self.uts_indices)
            np.random.shuffle(self.active_indices)
            return
        num_uts_needed = num_mts // ratio
        if num_uts_needed > len(self.uts_indices):
            selected_uts = np.array(self.uts_indices)
        else:
            selected_uts = np.random.choice(
                self.uts_indices, size=num_uts_needed, replace=False
            )
        self.active_indices = np.concatenate([self.mts_indices, selected_uts])
        np.random.shuffle(self.active_indices)

class SLIPTensorCollator:
    def __init__(self, tokenizer, max_len: int,):
        """
        Args:
            tokenizer: Hugging Face tokenizer
            max_len: maximum text length for truncation
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        # FIXME: Relax the hardcoded.
        self.sensor_max_len = 64 * 16
        self.sensor_max_nvar = 9

    def __call__(self, batch):
        """
        batch: list of dicts with keys
            - 'sensor': Tensor [C, L] or [1, L]
            - 'patch_size': int
            - 'caption': str
        Returns:
            dict of batched tensors, keeping sensor–caption alignment
        """
        sensors = []
        masks = []
        time_indicies = []
        lens = [] # find maximum number of tokens.
        nvars = [] # find maximum number of variables

        captions = [b['caption'] for b in batch]
        attn_masks = [b['attention_mask'] for b in batch if b['attention_mask'] is not None]

        for b in batch:
            sensor = b['sensor']  # nvar, L
            mask = b['mask'] # nvar, L
            time_index = b['time_index'] # nvar, L
            sensors.append(sensor)
            masks.append(mask)
            time_indicies.append(time_index)

            lens.append(sensor.shape[1])
            nvars.append(sensor.shape[0])

        sensors_padded, masks_padded, time_indicies_padded = self.sensor_processor(sensors, masks, time_indicies, lens, nvars)

        if self.tokenizer:
            tokenized = self.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )

            input_ids = tokenized['input_ids']
            attn_masks = tokenized['attention_mask']

        else:
            input_ids = None
            attn_masks = None


        return {
            'sensor': {
                'input_ids': sensors_padded, # BS, nvar, L
                'attention_mask': masks_padded, # BS, nvar, L
                'time_index': time_indicies_padded # BS, nvar, L
            },
            'text':
            {
                'input_ids': input_ids,             # [B, T]
                'attention_mask': attn_masks,   # [B, T]
                'description': captions                  
            }
        }

    def sensor_processor(self, sensors, masks, time_indicies, lens, nvars):
        """
        Process 2D tensors [nvar, length]
        """
        max_len_raw = int(np.max(lens))
        max_nvar_raw = int(np.max(nvars))

        # Clamp max dimensions to your hardcoded limits
        max_len = min(self.sensor_max_len, max_len_raw)
        max_nvar = min(self.sensor_max_nvar, max_nvar_raw)
        
        sensors_padded = []
        masks_padded = []
        time_indicies_padded = []

        for x, m, t in zip(sensors, masks, time_indicies):
            # x shape: [nvar, length]
            nvar, length = x.shape

            # 1. Truncate Time (dim 1)
            if length > max_len:
                x = x[:, :max_len]
                m = m[:, :max_len]
                t = t[:, :max_len]

            # 2. Truncate Variables (dim 0)
            if nvar > max_nvar:
                # Random choice for training
                selected_indices = np.random.choice(nvar, size=max_nvar, replace=False)
                x = x[selected_indices, :]
                t = t[selected_indices, :]
                m = m[selected_indices, :]

            # 3. Pad Time (dim 1)
            pad_len = max_len - x.shape[1]
            if pad_len > 0:
                # F.pad for 2D input: (pad_left, pad_right, pad_top, pad_bottom)
                # We want to pad right (end of sequence)
                pad_args = (0, pad_len) 
                x = F.pad(x, pad_args, value=0.0)
                m = F.pad(m, pad_args, value=0.0)
                t = F.pad(t, pad_args, value=0.0)

            # 4. Pad Variables (dim 0)
            pad_nvar = max_nvar - x.shape[0]
            if pad_nvar > 0:
                # We want to pad bottom (add rows/variables)
                # (left=0, right=0, top=0, bottom=pad_nvar)
                pad_args = (0, 0, 0, pad_nvar)
                x = F.pad(x, pad_args, value=0.0)
                m = F.pad(m, pad_args, value=0.0)
                t = F.pad(t, pad_args, value=0.0)

            sensors_padded.append(x)
            masks_padded.append(m)
            time_indicies_padded.append(t)

        sensors_padded = torch.stack(sensors_padded, dim=0)
        masks_padded = torch.stack(masks_padded, dim=0)
        time_indicies_padded = torch.stack(time_indicies_padded, dim=0)

        return sensors_padded, masks_padded, time_indicies_padded
    


class SLIPCollator:
    def __init__(self,
                  tokenizer, 
                  max_len: int,
                  sensor_max_len=384, 
                  sensor_max_nvar=6):
        """
        Args:
            tokenizer: Hugging Face tokenizer
            max_len: maximum text length for truncation
            sensor_max_len: maximum number of patches (time dimension) for sensor data
            sensor_max_nvar: maximum number of variables (channels) for sensor data
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.sensor_max_len = sensor_max_len 
        self.sensor_max_nvar = sensor_max_nvar

    def __call__(self, batch):
        """
        batch: list of dicts with keys
            - 'sensor': Tensor [C, L] or [L]
            - 'patch_size': int
            - 'caption': str
        Returns:
            dict of batched tensors, keeping sensor–caption alignment
        """
        sensors = []
        masks = []
        time_indicies = []
        lens = [] # find maximum number of tokens.
        nvars = [] # find maximum number of variables

        captions = [b['caption'] for b in batch]

        for b in batch:
            sensor = b['sensor']  # nvar, num_p, psize
            mask = b['mask'] # nvar, num_p, psize
            time_index = b['time_index'] # nvar, num_p, psize
            sensors.append(sensor)
            masks.append(mask)
            time_indicies.append(time_index)

            lens.append(sensor.shape[1])
            nvars.append(sensor.shape[0])

        sensors_padded, masks_padded, time_indicies_padded = self.sensor_processor(sensors, masks, time_indicies, lens, nvars)
        sensors_padded = [ [sensors_padded[i][j] for j in range(sensors_padded[i].shape[0])] for i in range(len(sensors_padded))]
        masks_padded = [ [masks_padded[i][j] for j in range(masks_padded[i].shape[0])] for i in range(len(masks_padded))]
        time_indicies_padded = [ [time_indicies_padded[i][j] for j in range(time_indicies_padded[i].shape[0])] for i in range(len(time_indicies_padded))]


        if self.tokenizer:
            tokenized = self.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )

            input_ids = tokenized['input_ids']
            attn_masks = tokenized['attention_mask']

        else:
            input_ids = None
            attn_masks = None


        return {
            'sensor': {
                'input_ids': sensors_padded, # list of list
                'attention_mask': masks_padded, # list of list
                'time_index': time_indicies_padded # list of list
            },
            'text':
            {
                'input_ids': input_ids,             # [B, T]
                'attention_mask': attn_masks,   # [B, T]
                'description': captions                  
            }
        }

    def sensor_processor(self,sensors, masks, time_indicies, lens, nvars):
        """
        preserve original behavior and thresholds
        sensors, masks, time_indicies: lists of tensors shaped [nvar, num_patches, psize]
        lens: list of num_patches per example
        nvars: list of nvar per example
        returns:
            sensors_padded: list of tensors padded and trimmed to final max_len and max_nvar
            masks_padded: list of corresponding masks
            time_indicies_padded: list of corresponding time indices
            max_len: final number of patches
            max_nvar: final number of variables
        """
        max_len_raw = int(np.max(lens))
        max_nvar_raw = int(np.max(nvars))

        max_len = min(self.sensor_max_len, max_len_raw)
        max_nvar = min(self.sensor_max_nvar, max_nvar_raw)
        
    
        sensors_padded = []
        masks_padded = []
        time_indicies_padded = []

        for x, m, t in zip(sensors, masks, time_indicies):
            # x, m: [nvar, num_patches, patch_size]
            nvar, num_p, psize = x.shape

            if num_p > max_len:
                x = x[:, :max_len, :]
                m = m[:, :max_len, :]
                t = t[:, :max_len, :]

            if nvar > max_nvar:
                # randomly select max_nvar variables
                selected_indices = np.random.choice(nvar, size=max_nvar, replace=False)
                x = x[selected_indices, :, :]
                t = t[selected_indices, :, :]
                m = m[selected_indices, :, :]

            # pad along patch dimension (dim=1)
            pad_patches = max_len - x.shape[1]
            if pad_patches > 0:
                x = F.pad(x, (0, 0, pad_patches, 0, 0, 0), value=0.0)
                m = F.pad(m, (0, 0, pad_patches, 0, 0, 0), value=0.0)
                t = F.pad(t, (0, 0, pad_patches, 0, 0, 0), value=0.0)

            # pad along variable dimension (dim=0)
            pad_nvar = max_nvar - x.shape[0]
            if pad_nvar > 0:
                x = F.pad(x, (0, 0, 0, 0, 0, pad_nvar), value=0.0)
                m = F.pad(m, (0, 0, 0, 0, 0, pad_nvar), value=0.0)
                t = F.pad(t, (0, 0, 0, 0, 0, pad_nvar), value=0.0)

            sensors_padded.append(x)
            masks_padded.append(m)
            time_indicies_padded.append(t)

        return sensors_padded, masks_padded, time_indicies_padded
    

class EvalDataset(Dataset):
    def __init__(self, data_dir,
                 is_train=True,
                 patch_size=16,
                 return_patch=True,
                 is_normalize=True,
                 hf_repo=None):

        self.return_patch = return_patch
        self.sensor_aug = None
        self.is_normalize = is_normalize
        self.patch_size = patch_size if patch_size is not None else 16

        local_exists = os.path.exists(f"{data_dir}/split_indices.json")

        if local_exists:
            with open(f"{data_dir}/split_indices.json", "r") as f:
                split_dict = json.load(f)

            self.data = np.load(f"{data_dir}/X.npy", mmap_mode="r")
            raw_labels = np.load(f"{data_dir}/Y.npy")

            if is_train == 'all':
                self.indicies = list(range(len(raw_labels)))
            else:
                self.indicies = split_dict['train'] if is_train else split_dict['test']

            label_map = {label: idx for idx, label in enumerate(np.unique(raw_labels))}
            self.labels = np.array([label_map[label] for label in raw_labels])

        else:
            from datasets import load_dataset
            task_name = os.path.basename(data_dir)
            split = 'train' if is_train else 'test'
            if is_train == 'all':
                # concatenate both splits
                ds_train = load_dataset(hf_repo, name=task_name, split='train')
                ds_test  = load_dataset(hf_repo, name=task_name, split='test')
                from datasets import concatenate_datasets
                ds = concatenate_datasets([ds_train, ds_test])
            else:
                ds = load_dataset(hf_repo, name=task_name, split=split)

            print(f"Local data not found at {data_dir}, loaded '{task_name}/{split}' from HuggingFace ({hf_repo})...")
            self.data = np.array(ds['X'], dtype=np.float32)
            self.indicies = list(range(len(ds)))
            self.labels = np.array(ds['label'], dtype=np.int64)

    def make_ce_weights(self):
        counts = np.bincount(self.labels)
        freq = counts / counts.sum()
        weights = 1.0 / freq
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, idx):
        real_idx = self.indicies[idx]
        sensor = self.data[real_idx].astype(np.float32)
        label = self.labels[real_idx]

        if self.is_normalize:
            sensor = sensor / (np.mean(np.abs(sensor) + 1e-6, axis=1, keepdims=True))

        sensor, mask, time_index = prepare_patches(sensor, self.patch_size, self.return_patch)

        return {
            'samples': sensor,
            'mask': mask,
            'time_index': time_index,
            'targets': torch.tensor(label, dtype=torch.long)
        }

class ZeroshotDataset(Dataset):
    def __init__(self,
                 data_dir,
                 sensor_aug=False,
                 patch_size=16, # default patch size
                 return_patch=True,
                 is_normalize=True,
                 hf_repo="LeoChen085/SlipDataset"):

        self.data_dir = data_dir
        self.return_patch = return_patch


        self.sensor_aug = None
        self.is_normalize = is_normalize

        local_exists = os.path.exists(f"{data_dir}/split_indices.json")

        if local_exists:
            with open(f"{data_dir}/split_indices.json", "r") as f:
                split_dict = json.load(f)
            self.indicies = split_dict['test']

            self.data = np.load(f"{data_dir}/X.npy", mmap_mode="r")

            # TODO: Constructing a int:text label map
            text_y = np.load(f"{data_dir}/text_y.npy")
            with open(f"{data_dir}/prompt.md", "r", encoding="utf-8") as f:
                prompt_template = Template(f.read())

            unique_text = np.unique(text_y)
            text_to_int = {t: i for i, t in enumerate(unique_text)}

            # integer labels aligned with text_y
            self.labels = np.array([text_to_int[t] for t in text_y], dtype=np.int64)

            # I need this to be revers s.t. int: text
            # build prompted text to int map
            self.label_map = {
                text_to_int[t]: prompt_template.substitute(label=t)
                for t in unique_text
            }
        else:
            # Load from HuggingFace
            task_name = os.path.basename(data_dir)
            print(f"Local data not found at {data_dir}, loading '{task_name}' from HuggingFace ({hf_repo})...")
            from datasets import load_dataset
            ds = load_dataset(hf_repo, name=task_name, split="test")

            self.data = np.array(ds['X'], dtype=np.float32)
            self.indicies = list(range(len(ds)))

            text_y = ds['text_label']
            prompt_template = Template(ds['prompt'][0])  # same value on every row

            unique_text = sorted(set(text_y))
            text_to_int = {t: i for i, t in enumerate(unique_text)}
            self.labels = np.array([text_to_int[t] for t in text_y], dtype=np.int64)
            self.label_map = {
                text_to_int[t]: prompt_template.substitute(label=t)
                for t in unique_text
            }

        self.patch_size = patch_size if patch_size is not None else 16

    def __len__(self):
        return len(self.indicies)
    
    def __getitem__(self, idx):
        real_idx = self.indicies[idx]
        sensor = self.data[real_idx].astype(np.float32) # (nvar, L)
        label = self.labels[real_idx]
        caption = self.label_map[label]

        # normalize
        if self.is_normalize:
            sensor = sensor / (np.mean(np.abs(sensor) + 1e-6, axis=1, keepdims=True))

        if self.sensor_aug is not None:
            sensor = self.sensor_aug(sensor)

        sensor,mask,time_index = prepare_patches(sensor, self.patch_size,self.return_patch)

        return {
            'sensor': sensor,
            'mask': mask,
            'time_index': time_index,
            'caption': caption,
            # 'targets': torch.tensor(label, dtype=torch.long)
            'patch_size': 16,
            'attention_mask': None
        }



from util.data_augmentation import TimeSeriesAugmentor
from sklearn.preprocessing import LabelEncoder

class SftDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split='train',
                 patch_size=16,
                 sensor_aug=True,
                 return_patch=True,
                 is_normalize=True,
                 hf_repo=None):

        local_path = os.path.join(data_dir, f"{split}.parquet")

        if os.path.exists(local_path):
            self.df = pd.read_parquet(local_path)
        else:
            from datasets import load_dataset
            hf_repo = hf_repo or "LeoChen085/SlipSFTDataset"
            task_name = os.path.basename(data_dir)   # e.g. "ecg_cot"
            print(f"Local data not found at {local_path}, loading '{task_name}/{split}' from HuggingFace ({hf_repo})...")
            ds = load_dataset(hf_repo, name=task_name, split=split)
            self.df = ds.to_pandas()

        try:
            answers = np.array(self.df["label"].tolist())
            le = LabelEncoder()
            self.encoded_answers = le.fit_transform(answers)
        except:
            self.encoded_answers = np.zeros(len(self.df), dtype=np.int64)

        self.patch_size = patch_size
        self.is_normalize = is_normalize
        self.return_patch = return_patch
        self.sensor_aug = TimeSeriesAugmentor() if sensor_aug else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encode_label = self.encoded_answers[idx]

        ts = np.stack(row['time_series'], axis=0).astype(np.float32)  # nvar, L

        if self.is_normalize:
            ts = ts / (np.mean(np.abs(ts) + 1e-6, axis=1, keepdims=True))

        if self.sensor_aug:
            ts = self.sensor_aug(ts)

        if self.patch_size is None:
            patch_size = get_patch_size(ts.shape[1])
            sensor, mask, time_index = prepare_patches(ts, patch_size, self.return_patch)
        elif isinstance(self.patch_size, (dict, DictConfig)):
            patch_size = int(self.patch_size.get(ts.shape[1]))
            sensor, mask, time_index = prepare_patches(ts, patch_size, self.return_patch)
        else:
            sensor, mask, time_index = prepare_patches(ts, self.patch_size, self.return_patch)

        return {
            "sample": sensor,
            "mask": mask,
            "time_index": time_index,
            "answer": row['answer'],
            "text_input": row['pre_prompt'] + row['post_prompt'],
            "encode_label": encode_label
        }


class SFTCollator:
    def __init__(self, tokenizer, max_len: int,is_test=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Hardcode tokenizer behavior
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.pad_token_id = self.tokenizer.pad_token_id
        self.is_test = is_test

    def __call__(self, batch,):
        sensors, sensor_masks, time_indices = [], [], []
        prompts, answers = [], []

        for item in batch:
            sensors.append(item["sample"])
            sensor_masks.append(item["mask"])
            time_indices.append(item["time_index"]) 
            prompts.append(item["text_input"].rstrip())
            answers.append(str(item["answer"]).strip())


        sensors_padded, masks_padded, time_indicies_padded = self.sensor_processor(sensors, sensor_masks, time_indices)
        sensors_padded = [ [sensors_padded[i][j] for j in range(sensors_padded[i].shape[0])] for i in range(len(sensors_padded))]
        masks_padded = [ [masks_padded[i][j] for j in range(masks_padded[i].shape[0])] for i in range(len(masks_padded))]
        time_indicies_padded = [ [time_indicies_padded[i][j] for j in range(time_indicies_padded[i].shape[0])] for i in range(len(time_indicies_padded))]


        if not self.is_test:
            # Training
            full_texts = [p + " " + a + self.tokenizer.eos_token for p, a in zip(prompts, answers)]
            tok = self.tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )

            input_ids = tok["input_ids"]
            attention_mask = tok["attention_mask"]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            for i in range(len(batch)):
                pad_len = int((attention_mask[i] == 0).sum().item())
                prefix = prompts[i].rstrip() + " "
                prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
                end = min(pad_len + len(prefix_ids), labels.size(1))
                labels[i, :end] = -100 
        else:
            prefix_prompt = [p for p in prompts]  # only use prompt for test
            labels = [a + self.tokenizer.eos_token for a in answers]  # separate labels for evaluation
            tok = self.tokenizer(
                prefix_prompt,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            
            input_ids = tok["input_ids"]
            attention_mask = tok["attention_mask"]

    

        return {
            "text": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            },
            "sensor": {
                "input_ids": sensors_padded,
                "attention_mask": masks_padded,
                "time_index": time_indicies_padded,
            },
        }

    def sensor_processor(self,sensors, masks, time_indicies):
        """
        preserve original behavior and thresholds
        sensors, masks, time_indicies: lists of tensors shaped [nvar, num_patches, psize]

        returns:
            sensors_padded: list of tensors left padded to max_len
            masks_padded: list of corresponding masks
            time_indicies_padded: list of corresponding time indices
            max_len: final number of patches
            max_nvar: final number of variables
        """
        lens = [x.shape[1] for x in sensors]
        nvars = [x.shape[0] for x in sensors]

        max_len = max(lens)
        max_nvar = max(nvars)
        
    
        sensors_padded = []
        masks_padded = []
        time_indicies_padded = []

        for x, m, t in zip(sensors, masks, time_indicies):
            # x, m: [nvar, num_patches, patch_size]
            nvar, num_p, psize = x.shape

            if num_p > max_len:
                x = x[:, :max_len, :]
                m = m[:, :max_len, :]
                t = t[:, :max_len, :]

            if nvar > max_nvar:
                # randomly select max_nvar variables
                selected_indices = np.random.choice(nvar, size=max_nvar, replace=False)
                x = x[selected_indices, :, :]
                t = t[selected_indices, :, :]
                m = m[selected_indices, :, :]

            # pad along patch dimension (dim=1)
            pad_patches = max_len - x.shape[1]
            if pad_patches > 0:
                x = F.pad(x, (0, 0, pad_patches, 0, 0, 0), value=0.0)
                m = F.pad(m, (0, 0, pad_patches, 0, 0, 0), value=0.0)
                t = F.pad(t, (0, 0, pad_patches, 0, 0, 0), value=0.0)

            # pad along variable dimension (dim=0)
            pad_nvar = max_nvar - x.shape[0]
            if pad_nvar > 0:
                x = F.pad(x, (0, 0, 0, 0, 0, pad_nvar), value=0.0)
                m = F.pad(m, (0, 0, 0, 0, 0, pad_nvar), value=0.0)
                t = F.pad(t, (0, 0, 0, 0, 0, pad_nvar), value=0.0)

            sensors_padded.append(x)
            masks_padded.append(m)
            time_indicies_padded.append(t)

        return sensors_padded, masks_padded, time_indicies_padded


class SFTTensorCollator:
    def __init__(self, tokenizer, max_len: int, sensor_max_len: int = 1024, sensor_max_nvar: int = 9):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # sensor_max_len corresponds to the total sequence length (patches * patch_size)
        self.sensor_max_len = sensor_max_len
        self.sensor_max_nvar = sensor_max_nvar

    def __call__(self, batch, is_test=False):
        sensors, sensor_masks, time_indices = [], [], []
        prompts, answers = [], []
        lens = []
        nvars = []

        for item in batch:
            # item["sample"] is [nvar, num_patches, psize]
            s = item["sample"]
            m = item["mask"]
            t = item["time_index"]

            # Flatten Patches and PatchSize into one dimension: [nvar, L]
            # This converts [3, 32, 4] -> [3, 128]
            nvar, num_p, psize = s.shape
            s = s.reshape(nvar, num_p * psize)
            m = m.reshape(nvar, num_p * psize)
            t = t.reshape(nvar, num_p * psize)

            sensors.append(s)
            sensor_masks.append(m)
            time_indices.append(t)
            
            lens.append(s.shape[1])
            nvars.append(s.shape[0])

            prompts.append(item["text_input"].rstrip())
            answers.append(str(item["answer"]).strip())

        # Process sensors into [Batch, Nvar, Max_Len]
        sensors_padded, masks_padded, time_indicies_padded = self.sensor_processor(
            sensors, sensor_masks, time_indices, lens, nvars
        )

        # Text processing
        if not is_test:
            # Training
            full_texts = [p + " " + a + self.tokenizer.eos_token for p, a in zip(prompts, answers)]
            tok = self.tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )

            input_ids = tok["input_ids"]
            attention_mask = tok["attention_mask"]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            for i in range(len(batch)):
                pad_len = int((attention_mask[i] == 0).sum().item())
                prefix = prompts[i].rstrip() + " "
                prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
                end = min(pad_len + len(prefix_ids), labels.size(1))
                labels[i, :end] = -100 
        else:
            prefix_prompt = [p + " " for p in prompts]  # only use prompt for test
            labels = [a + self.tokenizer.eos_token for a in answers]  # separate labels for evaluation
            tok = self.tokenizer(
                prefix_prompt,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            
            input_ids = tok["input_ids"]
            attention_mask = tok["attention_mask"]


        return {
            "text": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels, #list of string
            },
            "sensor": {
                "input_ids": sensors_padded,      # [B, nvar, L]
                "attention_mask": masks_padded,   # [B, nvar, L]
                "time_index": time_indicies_padded, # [B, nvar, L]
            },
        }

    def sensor_processor(self, sensors, masks, time_indicies, lens, nvars):
        """
        Process 2D tensors [nvar, length] to match UTSDTensorCollator
        """
        max_len_raw = int(np.max(lens))
        max_nvar_raw = int(np.max(nvars))

        max_len = min(self.sensor_max_len, max_len_raw)
        max_nvar = min(self.sensor_max_nvar, max_nvar_raw)
        
        sensors_padded = []
        masks_padded = []
        time_indicies_padded = []

        for x, m, t in zip(sensors, masks, time_indicies):
            # x shape: [nvar, length]
            nvar, length = x.shape

            # 1. Truncate Time
            if length > max_len:
                x = x[:, :max_len]
                m = m[:, :max_len]
                t = t[:, :max_len]

            # 2. Truncate Variables
            if nvar > max_nvar:
                selected_indices = np.random.choice(nvar, size=max_nvar, replace=False)
                x = x[selected_indices, :]
                m = m[selected_indices, :]
                t = t[selected_indices, :]

            # 3. Pad Time (Right Padding)
            pad_len = max_len - x.shape[1]
            if pad_len > 0:
                # For 2D [nvar, length], pad_args (left, right) pads the last dim
                pad_args = (0, pad_len) 
                x = F.pad(x, pad_args, value=0.0)
                m = F.pad(m, pad_args, value=0.0)
                t = F.pad(t, pad_args, value=0.0)

            # 4. Pad Variables (Bottom Padding)
            pad_nv = max_nvar - x.shape[0]
            if pad_nv > 0:
                # For 2D, (left, right, top, bottom)
                pad_args = (0, 0, 0, pad_nv)
                x = F.pad(x, pad_args, value=0.0)
                m = F.pad(m, pad_args, value=0.0)
                t = F.pad(t, pad_args, value=0.0)

            sensors_padded.append(x)
            masks_padded.append(m)
            time_indicies_padded.append(t)

        return torch.stack(sensors_padded), torch.stack(masks_padded), torch.stack(time_indicies_padded)
    


def get_patch_size(text_length):
    if text_length < 129:
        return 4
    elif text_length < 513:
        return 16
    elif text_length < 1025:
        return 32
    else:
        return 64



class ChatTsSftDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 split='train',
                 sensor_aug=True,
                 input_mode='ts' # ts, text, image
                 ):
        
        fn = os.path.join(data_dir, f"{split}.parquet")
        self.df = pd.read_parquet(fn)
        self.input_mode = input_mode
        try:
            answers = np.array(self.df["label"].tolist()) # single string label..
            le = LabelEncoder()
            self.encoded_answers = le.fit_transform(answers)
        except:
            # in case of missing label column
            self.encoded_answers = np.zeros(len(self.df), dtype=np.int64)


        if sensor_aug:
            self.sensor_aug = TimeSeriesAugmentor()
        else:
            self.sensor_aug = None

        
    def __len__(self):
        return len(self.df)
    
    def _ts_to_text(self, series,precision=2, return_meta=True):
        '''
        take in (L,) tensor, output string
        '''
        series = np.array(series, dtype=float)
        
        # apply min-max and record the scale and offset, 
        min_val, max_val = np.min(series), np.max(series)
        scale = max_val - min_val
        offset = min_val

        if scale < 1e-6:
            series = np.ones_like(series)
            scale = max_val
        else:
            series = (series - offset) / scale


        # serialize
        encoded = serialize_arr(series, SerializerSettings(prec=precision))
        
        if return_meta:
            return encoded, {"scale": np.round(scale, precision), "offset": np.round(offset, precision), 'precision': precision}
        
        return encoded, {}


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encode_label = self.encoded_answers[idx]
        answer = row['answer'] # cot answer

        ts = np.stack(row['time_series'], axis=0).astype(np.float32) # nvar, L
        # chatts normalize handle in processor already
        
        if self.sensor_aug:
            ts = self.sensor_aug(ts)
        
        nvar = ts.shape[0]
        prompt = ''
        for i in range(nvar):
            prompt += f"I have time series {i}: <ts><ts/>."

        prompt = prompt + row['pre_prompt'] + row['post_prompt']
        # prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        if self.input_mode == 'text':
            offset_template = Template("[offset: $offset, scale: $scale, precision: $precision]: ")
            # seralize it
            text_ts = []
            for i in range(ts.shape[0]):
                encoded, meta = self._ts_to_text(ts[i])
                offset_str = offset_template.substitute(
                    offset = meta['offset'],
                    scale = meta['scale'],
                    precision = meta['precision']
                )
                ts_description = offset_str + encoded
                text_ts.append(ts_description)
            
            # check number of <ts><ts/> matches number of text_ts, then do a exchange in order.
            n_tokens = prompt.count('<ts><ts/>')
            if n_tokens != len(text_ts):
                raise ValueError(
                    f"Mismatch between placeholders and serialized series. "
                    f"placeholders={n_tokens}, serialized={len(text_ts)}"
                )
            
            for s in text_ts:
                prompt = prompt.replace('<ts><ts/>', s, 1)

            # print(prompt)

        elif self.input_mode == 'image':
            # convert to image
            pass
        else:
            # default ts
            pass

        return {
            "sample": ts,
            # "mask": mask,
            # "time_index": time_index,
            "answer": answer, # sentence answer
            "text_input": prompt,
            "encode_label": encode_label
        }
    
def ChatTsSftcollator(
                    processor, 
                    batch, 
                    input_mode='ts',
                    is_test=False
                    ):

    # flatten all the time series into one list
    ts_list = []
    formatted_prompts = []
    answer = []
    for item in batch:
        ts_list.extend(item['sample'])
        formatted_prompts.append(item['text_input'])
        answer.append(item['answer'])


    if input_mode == 'ts':
        if not is_test:
            full_texts = [p + " " + a + processor.tokenizer.eos_token for p, a in zip(formatted_prompts, answer)]
            tok = processor(text=full_texts, 
                                timeseries=ts_list, 
                                padding=True, 
                                return_tensors="pt")
            
            input_ids = tok["input_ids"]
            attention_mask = tok["attention_mask"]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            for i in range(len(batch)):
                pad_len = int((attention_mask[i] == 0).sum().item())
                prefix = formatted_prompts[i].rstrip() + " "
                prefix_ids = processor(text=prefix, 
                                    timeseries=[ts_list[i]],
                                    add_special_tokens=False)["input_ids"]
                
                end = min(pad_len + len(prefix_ids), labels.size(1))
                labels[i, :end] = -100
            
            timeseries = tok['timeseries']
        else:
            prefix_prompt = [p for p in formatted_prompts]  # only use prompt for test
            labels = [a + processor.tokenizer.eos_token for a in answer]  # separate labels for evaluation
            tok = processor(
                text = prefix_prompt,
                timeseries=ts_list,
                padding=True,
                return_tensors="pt",
            )
            
            input_ids = tok["input_ids"]
            attention_mask = tok["attention_mask"]
            timeseries = tok['timeseries']

    else:
        # raise error
        raise ValueError(f"Did not Implement")

    
    return {
        "text":{
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'timeseries': timeseries,
            "labels": labels,
        },
        "sensor": None
    }
    



    


    
if __name__ == "__main__":
    pass
