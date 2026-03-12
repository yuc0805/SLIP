'''
Modified from PatchTST: https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_self_supervised/src/models/patchTST.py
'''

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange
import contextlib

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    '''
    t: B, L, D
    mask: B, L, 1
    '''
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high   
        # self.low, self.high = ranges        
    def forward(self, x):                    
        # return sigmoid_range(x, self.low, self.high)
        return torch.sigmoid(x) * (self.high - self.low) + self.low


class RegressionHead(nn.Module):
    def __init__(self, 
                 d_model, 
                 output_dim,
                 head_dropout=0.1, 
                 y_range=None): # tuple (min, max)
        super().__init__()
        self.y_range = y_range
        self.in_features = d_model
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.linear.weight, std=2e-5)


    def forward(self, x):
        """
        x: bs x d_model
        output: bs x output_dim
        """

        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: 
            y = SigmoidRange(*self.y_range)(y)  

        return y


class ClassificationHead(nn.Module):
    def __init__(self, 
                 d_model,
                 n_classes,
                 head_dropout=0.1):
        super().__init__()
        
        self.in_features = d_model
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, n_classes)
        
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.linear.weight, std=2e-5)

    def forward(self, x):
        """
        x: [bs x d_model]
        output: [bs x n_classes]
        """
       
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes

        return y
    

class MultimodalClassificationWrapper(nn.Module):
    def __init__(self, 
                 model, 
                 head_type,
                 num_classes,
                 y_range=None,):
        super().__init__()
        """
        Args:
            model: multimodal encoder that gives a multimodal representation
        """
  
        self.model = model
        d_model = model.lm_dim
    
        self.num_classes = num_classes
        # create task-specific heads

        if head_type == "regression":
            self.head = RegressionHead(
                d_model=d_model,
                output_dim=num_classes,
                head_dropout=0.1,
                y_range=y_range
            )
        elif head_type == "classification":
            self.head = ClassificationHead(
                d_model=d_model,
                n_classes=num_classes,
                head_dropout=0.1
            )
        else:
            raise ValueError(f"Unknown head_type {head_type}")

    def forward(self, question, images):
        """
        Args:
            question: input tensor for the question
            images: input tensor for the images
            mask: optional padding mask for the backbone (patch embed)
        Returns:
            dict: {task_name: prediction tensor}
        """

        x = self.model.get_multimodal_feature(question, images)  # (B, embed_dim)  
        out = self.head(x).squeeze(-1)  # (B, num_classes) or (B, output_dim)

        return out


class ClassificationWrapper(nn.Module):
    def __init__(self, 
                 model, 
                 head_type,
                 num_classes,
                 y_range=None,
                 glob_pool='avg'):
        super().__init__()
        """
        Args:
            model: backbone encoder (e.g. ViT)
        """
        self.model = model
    
        self.glob_pool = glob_pool
        d_model = model.embed_dim
        self.num_classes = num_classes
        # create task-specific heads

        if head_type == "regression":
            self.head = RegressionHead(
                d_model=d_model,
                output_dim=num_classes,
                head_dropout=0.1,
                y_range=y_range
            )
        elif head_type == "classification":
            self.head = ClassificationHead(
                d_model=d_model,
                n_classes=num_classes,
                head_dropout=0.1
            )
        else:
            raise ValueError(f"Unknown head_type {head_type}")

    def forward(self, x, mask=None,time_index=None):
        """
        Args:
            x: input tensor for the backbone
            mask: optional padding mask for the backbone (patch embed)
        Returns:
            dict: {task_name: prediction tensor}
        """
        BS = len(x)
        x,attn_mask = self.model(x,mask,time_index)  # (B, num_patches+1, embed_dim)
      
        # global pooling
        if self.glob_pool == 'avg':
            if attn_mask is not None:
                attn_mask = rearrange(attn_mask, 'b n p -> b (n p) 1')  
                x = masked_mean(x,attn_mask,dim=1)
            else:
                x = x.mean(dim=1)
                
        elif self.glob_pool == 'cls':
            x = x[:, 0]  # use cls token
        else:
            raise NotImplementedError
        
        out = self.head(x)  # (B, num_classes) or (B, output_dim)

        return out

from transformers.activations import ACT2FN
import torch.nn.functional as F

class MultiSizeProjHead(nn.Module):
    def __init__(self, out_dim, base_patch=32, **cfg):
        super().__init__()

        '''
        out_dim is the dim that the model is predicting,
        if pixel-regression, out_dim = 1
        if quantil-regression, out_dim = num_quantiles (21)
        '''
        self.base_patch = base_patch
        self.out_dim = out_dim

        hidden_size = cfg['embed_dim']
        intermediate_size = cfg['mlp_ratio'] * hidden_size 
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.out_dim_base = base_patch * out_dim

        self.input_proj = nn.Linear(hidden_size, intermediate_size)
        self.shared_linear = nn.Linear(intermediate_size, self.out_dim_base)
        self.shared_residual = nn.Linear(hidden_size, self.out_dim_base)
        self.dropout = nn.Dropout(0.1)
        self.act = ACT2FN['silu']

        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def resize_weight(self, out_patch_size):
        """
        Interpolate weights along the output dimension to target patch size.
        Target output dim = out_patch_size * out_dim
        """
        target_dim = out_patch_size * self.out_dim

        base_w = self.shared_linear.weight       # [out_dim_base, intermediate]
        base_b = self.shared_linear.bias
        
        res_w = self.shared_residual.weight      # [out_dim_base, hidden]
        res_b = self.shared_residual.bias

        new_w = F.interpolate(
            base_w.t().unsqueeze(0), 
            size=target_dim, 
            mode="linear", 
            align_corners=False
        )[0].t().to(base_w.dtype)
        
        # Bias is [Out], needs resizing too
        new_b = None
        if base_b is not None:
            new_b = F.interpolate(
                base_b.view(1, 1, -1), 
                size=target_dim, 
                mode="linear", 
                align_corners=False
            ).view(-1).to(base_b.dtype)

        new_res_w = F.interpolate(
            res_w.t().unsqueeze(0),
            size=target_dim, 
            mode="linear", 
            align_corners=False
        )[0].t().to(res_w.dtype)
        
        new_res_b = None
        if res_b is not None:
            new_res_b = F.interpolate(
                res_b.view(1, 1, -1), 
                size=target_dim, 
                mode="linear", 
                align_corners=False
            ).view(-1).to(res_b.dtype)

        return new_w, new_b, new_res_w, new_res_b

    def forward(self, x, patch_sizes):
        """
        x: tensor of shape (batch * nvar, num_patches, hidden_size)
        patch_sizes: tensor of shape (batch * nvar) integers

        Returns:
            list of tensors, where element i has shape (num_patches, patch_sizes[i] * num_quantiles)
        """
        # amp_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
        # device = x.device

        xs = self.input_proj(x) # [B, num_p, intermediate]

        unique_sizes = patch_sizes.unique(sorted=True)
        N = x.shape[0] # batch size
        outputs_list = [None] * N

        for psize in unique_sizes.tolist():
            idxs = (patch_sizes == psize).nonzero(as_tuple=True)[0]
            
            # Gather data for this patch size
            xs_group = xs[idxs]      # [B_g, num_p, intermediate]
            x_res_group = x[idxs]    # [B_g, num_p, hidden]

            # Resize weights
            w, b, r_w, r_b = self.resize_weight(psize)
            
            lin_out = F.linear(xs_group, w, b)
            res_out = F.linear(x_res_group, r_w, r_b)

            hid = self.act(lin_out)
            hid = self.dropout(hid)
            
            final_group = hid + res_out # [B_g, num_p, psize*quantiles]

            # Scatter results back to the list
            # We unbind the batch dimension to get individual tensors
            for i, result_tensor in zip(idxs.tolist(), final_group.unbind(0)):
                outputs_list[i] = result_tensor

        return outputs_list