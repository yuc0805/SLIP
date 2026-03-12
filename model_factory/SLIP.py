# Reference: https://github.com/lucidrains/CoCa-pytorch/blob/main/coca_pytorch/coca_pytorch.py

import math

from sympy import shape
from omegaconf import DictConfig
import torch
torch._dynamo.config.capture_scalar_outputs = True
from torch import Tensor, einsum, nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.distributed as dist
from einops import rearrange, repeat,reduce
from model_factory.multimodal_gemma import Gemma3MultimodalModel
import hydra
# for generation
from typing import Optional, List, Union
import contextlib
from transformers.generation.utils import GenerationMixin
from model_factory.ts_transformer import AttentionPooling

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    '''
    t: B, L, D
    mask: B, L, 1
    '''
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

# helper metric: https://arxiv.org/pdf/2005.10242
def lalign(x, y, alpha=2):
    # calculate the closness of the positive pairs.
    return (x - y).norm(dim=1).pow(alpha).mean()

def lunif(x, t=2):
    # calculate the uniformity of one side.
    sq = torch.pdist(x, p=2).pow(2)
    return sq.mul(-t).exp().mean().log()

# distributed
def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

# https://huggingface.co/Qwen/Qwen3-Embedding-8B
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def all_gather_variable_batch(x):
    """
    All-gather variable sized tensors across DDP ranks.
    x: [B_local, D]
    Returns:
        out: [sum(B_local across ranks), D]
        sizes: python list of sizes per rank
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    device = x.device

    # 1. Gather sizes
    local_size = torch.tensor([x.shape[0]], device=device, dtype=torch.long)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world)]
    dist.all_gather(all_sizes, local_size)
    sizes = [int(s.item()) for s in all_sizes]

    # 2. Pad local tensor to max size
    max_size = max(sizes)
    if local_size < max_size:
        pad_len = max_size - local_size
        padding = torch.zeros(pad_len, *x.shape[1:], device=device, dtype=x.dtype)
        x_padded = torch.cat([x, padding], dim=0)
    else:
        x_padded = x

    # 3. All-gather padded tensors
    gathered = [torch.zeros_like(x_padded) for _ in range(world)]
    dist.all_gather(gathered, x_padded)

    # 4. Trim each rank's padded slice
    trimmed = [g[:sizes[i]] for i, g in enumerate(gathered)]

    # 5. Concatenate true global batch
    out = torch.cat(trimmed, dim=0)
    return out, sizes

class AllGather(Function):
    @staticmethod
    def forward(ctx, x):
        assert dist.is_initialized() and dist.get_world_size() > 1
        x, batch_sizes = all_gather_variable_batch(x)
        ctx.batch_sizes = batch_sizes
        return x

    @staticmethod
    def backward(ctx, grads):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = 0)
        return grads_by_rank[rank]

all_gather = AllGather.apply


# to latents
class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)



class SLIP(nn.Module,GenerationMixin):
    _is_stateful = False 
    def __init__(
        self,
        tokenizer=None, #legacy argument.
        **kwargs
    ):
        super().__init__()

        self.tokenizer = tokenizer
        enc_cfg = kwargs['sensor_encoder_cfg']
        if isinstance(enc_cfg, (DictConfig, dict)):
            self.sensor_encoder = hydra.utils.instantiate(enc_cfg)
        else:
            self.sensor_encoder = enc_cfg

        
        ############################################################
        dim = self.sensor_encoder.embed_dim # 384
        text_encoder = kwargs['llm_model_name']
        self.embed_dim = dim
        self.use_lora = kwargs.get('use_lora', True)
        self.post_train = kwargs.get('post_train', True)
        self.use_sig_loss = kwargs.get('use_sig_loss', False)
        ##########################################

        ## Text encoder ####
        self.caption_loss_weight = kwargs['caption_loss_weight'] 
        self.max_llm_len = kwargs['max_llm_len']
        self.multimodalModel = Gemma3MultimodalModel(text_encoder,self.post_train)

        if self.caption_loss_weight <= 0:
            self.multimodalModel._truncate_to_unimodal()
            
        unlocked_layers = kwargs.get('unlocked_layers', 0)
        if unlocked_layers < 12: # 12 is the split layer
            self.multimodalModel._lock_text(
                unlocked_layers=unlocked_layers,
                freeze_layer_norm=kwargs.get('freeze_layer_norm', True)
            )
            
        lm_dim = self.multimodalModel.hidden_size #640
        self.lm_dim = lm_dim
        common_dim = lm_dim # harcoded for now
        # self.multimodalModel.model.gradient_checkpointing_enable()
        #########################################
        
        num_img_queries = kwargs.get('num_img_queries', 0)
        if num_img_queries>0:
            self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, common_dim))
            self.img_attn_pool = AttentionPooling(
                dim=common_dim,
                context_dim=dim,
                num_heads=kwargs['num_heads']) # pre-norm+post_norm

            dim = common_dim
            
        # normalize.
        self.img_to_latents = EmbedToLatents(dim, common_dim)
        self.text_to_latents = EmbedToLatents(common_dim, common_dim)
        

        # learnable temperature
        self.temperature = nn.Parameter(torch.tensor(math.log(1/0.07)))
        self.temperature_max = math.log(1/0.07)
        if self.use_sig_loss:
            # default implementation
            self.temperature = nn.Parameter(torch.tensor(math.log(10)))
            #self.temperature_max = math.log(10)
            self.temperature_max = 999 # trivally large, so no upper bound.
            self.logit_bias = nn.Parameter(torch.ones([]) * -10) 


        # multimodal decoder #############
        pad_token_id = self.tokenizer.pad_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.contrastive_loss_weight = kwargs['contrastive_loss_weight']  
        ##################################
            
        self._init_weights()
        # whether in data parallel setting
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"TRAINABLE: {name}")


    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # apply only to modules we added
        self.img_to_latents.apply(_init)
        self.text_to_latents.apply(_init)

        if hasattr(self, 'img_attn_pool'):
            self.img_attn_pool.apply(_init)
            nn.init.xavier_uniform_(self.img_queries)

    def get_lora_parameters(self): # make training script happy
        """
        Gathers:
        1. LoRA weights (A and B matrices) inside Gemma.
        2. Full-parameter updated 'modules_to_save' (Embeddings/Head).
        3. Full-parameter updated Cross-Attention blocks.
        4. Bridge layers (img_to_latents, text_to_latents, etc.).
        """
        if not self.use_lora:
            return []
        
        trainable_params = []
        
        # 1. Check the multimodal LLM (Gemma + LoRA + Cross-Attn)
        for name, param in self.multimodalModel.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # 2. Check the Bridge modules
        bridge_modules = [self.img_to_latents, self.text_to_latents]
        if hasattr(self, 'img_attn_pool'):
            bridge_modules.append(self.img_attn_pool)
            
        for module in bridge_modules:
            for param in module.parameters():
                if param.requires_grad:
                    trainable_params.append(param)
                    
        # 3. Check the Queries and Sensor Encoder
        if hasattr(self, 'img_queries') and self.img_queries.requires_grad:
            trainable_params.append(self.img_queries)
        
        # Optionally add sensor_encoder if you haven't locked it
        for param in self.sensor_encoder.parameters():
            if param.requires_grad:
                trainable_params.append(param)
            
        return trainable_params
    
    def _pad_to_len(self, x, max_len):
        # pad along dim 1 to max_len with zeros
        if x.dim() == 3:
            # [B, L, D]
            pad_len = max_len - x.size(1)
            if pad_len > 0:
                pad = x.new_zeros(x.size(0), pad_len, x.size(2))
                x = torch.cat([pad, x], dim=1)

        elif x.dim() == 2:
            # [B, L] case such as masks
            pad_len = max_len - x.size(1)
            if pad_len > 0:
                pad = x.new_zeros(x.size(0), pad_len)
                x = torch.cat([pad, x], dim=1)
        return x

    def _gather_features(self, img, txt, gather_with_grad=False):
        """Return all features if DDP, else inputs. Same batch size per rank assumed."""
        if not (dist.is_available() and dist.is_initialized()):
            return img, txt
        
        ### prepare for gathering ###
        #
        # get max length across ranks for padding.
        img_len = torch.tensor([img.size(1)], device=img.device, dtype=torch.long)
        txt_len = torch.tensor([txt.size(1)], device=txt.device, dtype=torch.long)
        dist.all_reduce(img_len, op=dist.ReduceOp.MAX)
        dist.all_reduce(txt_len, op=dist.ReduceOp.MAX)
        max_img_len = int(img_len.item())
        max_txt_len = int(txt_len.item())

        img = self._pad_to_len(img, max_img_len)
        txt = self._pad_to_len(txt, max_txt_len)
        #################################

        if gather_with_grad:
            # keep grad across ranks
            all_img = all_gather(img)
            all_txt = all_gather(txt)
        else:
            # no grad path, saves memory
            ws = dist.get_world_size()
            outs_i = [torch.empty_like(img) for _ in range(ws)]
            outs_t = [torch.empty_like(txt) for _ in range(ws)]

            try:
                dist.all_gather(outs_i, img.contiguous())
                dist.all_gather(outs_t, txt.contiguous())
                    
            except Exception as e:
                print("Error occurred while gathering features:", e)
                
            outs_i[dist.get_rank()] = img
            outs_t[dist.get_rank()] = txt
            all_img = torch.cat(outs_i, dim=0)
            all_txt = torch.cat(outs_t, dim=0)

        return all_img, all_txt

    def embed_text(self, 
                   input_ids,
                   attention_mask,
                   text_embed=None):
        '''
        need to make this casual to avoid representation leak.

        text: (BS, llm_seq_len) token_ids
        attn_mask: (Bs, llm_seq_len)
        '''

        if text_embed is not None:
            hidden_states = text_embed # (BS, max_seq_len, lm_dim)
            
        else:
            outputs = self.llm(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=False, # Set to False or remove
                # use_cache=False             # Ensure cache is off for training/gradient ckpt
            )

            hidden_states = outputs.last_hidden_state
        
        return hidden_states
        
        

    def embed_sensor(self, sensors, sensor_attn_mask=None, time_index=None):
        '''
        sensors: (BS, num_channels, L)
        '''
        
        sensor_tokens, attn_mask = self.sensor_encoder(sensors, sensor_attn_mask, time_index=time_index) 
        # sensor_tokens: Bs,(nvar, num_p), img_dim
        # attn_mask: BS, nvar, num_p
        
        if hasattr(self, 'img_attn_pool'):
            img_queries = repeat(self.img_queries, 'n d -> b n d', b=sensor_tokens.shape[0])
            sensor_tokens = self.img_attn_pool(img_queries, sensor_tokens,attn_mask)
        
        return sensor_tokens, attn_mask.bool()

    # use an openCLIP implementation
    def forward_loss(self, 
                     text_hidden, 
                     sensor_hidden, 
                     sensor_mask,
                     gather_with_grad=False):

        '''
        text_embd: tuple of (text_cls, text_tokens)
        sensor_embed: tuple of (sensor_cls, sensor_tokens)
        sensor_mask: (BS, nvar, num_p)
        '''
        
        # global features
        if hasattr(self, 'img_attn_pool'):
            # use cls token
            sensor_hidden = sensor_hidden[:, 0, :]
        else:
            sensor_hidden = masked_mean(sensor_hidden, rearrange(sensor_mask, 'b n p -> b (n p) 1'), dim=1)  # BS, img_dim
        
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        if world > 1:
            all_img, all_txt = self._gather_features(sensor_hidden, text_hidden, gather_with_grad=gather_with_grad)
        else:
            all_img, all_txt = sensor_hidden, text_hidden
        

        contrastive_loss = self.CLIP_loss(all_txt, all_img)*self.contrastive_loss_weight

        # some supplementry losses
        align_loss = lalign(all_txt, all_img)
        unif_txt = lunif(all_txt)
        unif_img = lunif(all_img)

        outputs = {
            "loss": contrastive_loss,
            'contrastive_loss': contrastive_loss,
            "align_loss": align_loss,
            "unif_txt": unif_txt,
            "unif_img": unif_img,
        }

        return outputs

    
    def CLIP_loss(
            self,
            text_cls,
            sensor_cls,):
        
        temperature = (self.temperature.clamp(max=self.temperature_max)).exp()
        logits_t2i = temperature * (text_cls @ sensor_cls.t())  # [B_global, B_global]
        targets = torch.arange(logits_t2i.size(0), device=sensor_cls.device)
        contrastive_loss = 0.5 * (
            F.cross_entropy(logits_t2i, targets) +
            F.cross_entropy(logits_t2i.t(), targets)
        )
        
        return contrastive_loss

    def sig_loss(self, text_hidden, sensor_hidden, sensor_mask):
        '''
        SigLip Loss: Decoupling contrastive-loss with batch size
        text_hidden: (BS, dim)
        sensor_hidden: (BS, sensor_len, dim)
        text_mask: (BS, text_len)
        sensor_mask: (BS, sensor_len)
        '''

        if hasattr(self, 'img_attn_pool'):
            # use cls token
            sensor_hidden = sensor_hidden[:, 0, :]
        else:
            sensor_hidden = masked_mean(sensor_hidden, rearrange(sensor_mask, 'b n p -> b (n p) 1'), dim=1)  # BS, img_dim

      
        logit_scale = self.temperature.clamp(max=self.temperature_max).exp()
        loss = self._sig_loss(sensor_hidden, text_hidden, logit_scale, self.logit_bias)

        return {'loss': loss, 'contrastive_loss': loss}


    def forward(
        self,
        text,
        sensors,
        prompt=None, # legacy input
        return_embeddings=False,
    ):


        sensor_hidden, sensor_mask = self.embed_sensor(sensors=sensors['input_ids'],
                                                   sensor_attn_mask=sensors['attention_mask'], # this is pixel-level mask
                                                   time_index=sensors['time_index'])
        
        # sensor_hidden: (BS, num_sensor_token, dim)
        self.multimodalModel.condition_image(sensor_hidden)
        text_hidden, logits = self.multimodalModel(input_ids=text['input_ids'][:,:-1],
                                                   attention_mask=text['attention_mask'][:,:-1], )
        # text_sentence_embed: (BS, dim)
        # logits: (BS, pred_len, vocab_size)

        labels = text['input_ids'][:,1:] # bs, pred_len
        #logits = rearrange(logits, 'b n c -> b c n') # bs, vocab_size, pred_len

        text_hidden = self.text_to_latents(text_hidden)
        sensor_hidden = self.img_to_latents(sensor_hidden)
        
        if self.use_sig_loss:
            loss_dict = self.sig_loss(text_hidden,
                                      sensor_hidden,
                                      sensor_mask)
        else:
            # This branch will need all-gather.
            loss_dict = self.forward_loss(text_hidden,
                                        sensor_hidden,
                                        sensor_mask,)

        
        if self.caption_loss_weight > 0:
            loss_logits = logits.reshape(-1, logits.size(-1)) # Shape: [BS * Seq, Vocab]
            loss_labels = labels.reshape(-1)                   # Shape: [BS * Seq]
            caption_loss = self.ce(loss_logits, loss_labels) * self.caption_loss_weight

            loss_dict['caption_loss'] = caption_loss
            loss_dict['loss'] = loss_dict['contrastive_loss'] + caption_loss
                                    

        return loss_dict
    
    def _lock_sensor(self,):
        # Freeze all sensor-related parameters (cross-attn blocks)
        for name, param in self.sensor_encoder.named_parameters():
            param.requires_grad = False

    def sft_training(self,text,sensors,return_output=False):
        sensor_hidden, _ = self.embed_sensor(sensors=sensors['input_ids'],
                                                   sensor_attn_mask=sensors['attention_mask'], 
                                                   time_index=sensors['time_index'])
        
        # sensor_hidden: (BS, num_sensor_token, dim)
        self.multimodalModel.condition_image(sensor_hidden)

        # debugging code.
        # sample_text = text['input_ids'][0]
        # sample_label = text['labels'][0]
        # # make the -100 to be the pad token id for decoding
        # sample_label = torch.where(sample_label==-100, self.tokenizer.pad_token_id, sample_label)
        # print('sample text:', self.tokenizer.decode(sample_text))
        # print('sample label:', self.tokenizer.decode(sample_label))
        # exit()
            

        outputs = self.multimodalModel.model(input_ids=text['input_ids'],
                                            attention_mask=text['attention_mask'],
                                            return_dict=True,)
                                                #    labels=text['labels'], )
        if return_output:
            return outputs
        
        logits = outputs.logits # (BS, pred_len, vocab_size)
        labels = text['labels'] # (BS, pred_len)
        # shift for causal lm
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # flatten logits for efficiency
        logss_logits = shift_logits.view(-1, shift_logits.size(-1)) # Shape: [BS * Seq, Vocab]
        loss_labels = shift_labels.view(-1)                   # Shape: [BS * Seq]

        # define a new loss for stf
        ce = torch.nn.functional.cross_entropy(
            logss_logits,
            loss_labels,
            reduction='none',
            ignore_index=-100,
        )

        if 'loss_weights' in text:
            loss_weights = text['loss_weights']
            loss_weights = loss_weights[:,1:].contiguous()
            loss_weights = loss_weights.view(-1)  # Shape: [BS * Seq]

            # apply weights
            weighted_ce = ce * loss_weights
            loss = weighted_ce.sum() / loss_weights.sum()
            
        else:
            loss = ce.mean()
        
        return {'loss': loss}

    def generate(self, 
                 text, 
                 sensors, 
                 **generate_kwargs):
        """
        Generates text conditioned on image embeddings.
        """

        sensor_hidden, _ = self.embed_sensor(sensors=sensors['input_ids'],
                                                   sensor_attn_mask=sensors['attention_mask'], # this is pixel-level mask
                                                   time_index=sensors['time_index'])
        
        self.multimodalModel.condition_image(sensor_hidden)

        generated_text = self.multimodalModel.model.generate(
            input_ids=text['input_ids'],
            attention_mask=text['attention_mask'],
            max_new_tokens=300,
            do_sample=False,
            num_beams=1,
            early_stopping=False,
        )

        return generated_text
    
        
    @ torch.no_grad()
    def get_embedding(self,text,sensors):
        sensor_hidden, sensor_mask = self.embed_sensor(sensors=sensors['input_ids'],
                                                   sensor_attn_mask=sensors['attention_mask'], # this is pixel-level mask
                                                    time_index=sensors['time_index'])
        
        self.multimodalModel.condition_image(sensor_hidden)
        text_hidden, _ = self.multimodalModel(input_ids=text['input_ids'][:,:-1],
                                                   attention_mask=text['attention_mask'][:,:-1], )
        
        text_hidden = self.text_to_latents(text_hidden)
        sensor_hidden = self.img_to_latents(sensor_hidden)

        if hasattr(self, 'img_attn_pool'):
            # use cls token
            sensor_hidden = sensor_hidden[:, 0, :]
        else:
            sensor_hidden = masked_mean(sensor_hidden, rearrange(sensor_mask, 'b n p -> b (n p) 1'), dim=1)  # BS, img_dim # (BS, dim)

        return text_hidden, sensor_hidden
    
    @ torch.no_grad()
    def get_sensor_embedding(self,input_ids,mask,time_index):
        sensor_hidden, sensor_mask = self.embed_sensor(sensors=input_ids,
                                                    sensor_attn_mask=mask, 
                                                    time_index=time_index)
        sensor_hidden = self.img_to_latents(sensor_hidden)

        if hasattr(self, 'img_attn_pool'):
            # use cls token
            sensor_hidden = sensor_hidden[:, 0, :]
        else:
            sensor_hidden = masked_mean(sensor_hidden, rearrange(sensor_mask, 'b n p -> b (n p) 1'), dim=1)  # BS, img_dim

        return sensor_hidden
    
    @ torch.no_grad()
    def get_text_embedding(self,text):
        text_mask = text['attention_mask']
        text_hidden = self.embed_text(text['input_ids'],
                                      attention_mask=text_mask,)

        text_hidden = self.text_to_latents(text_hidden)
    
        if self.llm.config.pooler == 'mean':
            text_hidden = masked_mean(text_hidden, rearrange(text_mask, 'b l -> b l 1').bool(), dim=1)  # BS, lm_dim
        else:
            text_hidden = last_token_pool(text_hidden, text_mask) # (BS, dim)

        return text_hidden
    
    def get_multimodal_feature(self, question, sensors):
        sensor_hidden, sensor_mask = self.embed_sensor(sensors=sensors['input_ids'],
                                                   sensor_attn_mask=sensors['attention_mask'], # this is pixel-level mask
                                                    time_index=sensors['time_index'])
        
        # sensor_hidden: (BS, num_sensor_token, dim)
        self.multimodalModel.condition_image(sensor_hidden)
        outputs = self.multimodalModel(input_ids=question['input_ids'],
                                                   attention_mask=question['attention_mask'],
                                                   return_embeddings=True)
        # text_sentence_embed: (BS, dim)
        # logits: (BS, pred_len, vocab_size)
        multimodal_hidden = outputs.hidden_states[-1][:,-1,:] # (BS, dim)

        return multimodal_hidden

                                      


class Config(dict):
    def __getattr__(self, key):
        return self[key]


