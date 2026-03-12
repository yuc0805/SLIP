# Reference: https://huggingface.co/thuml/sundial-base-128m/blob/main/modeling_sundial.py

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from util.pos_embed import RotaryEmbedding, apply_rotary_pos_emb,apply_rotary_pos_emb_2d, build_2d_position_ids
from transformers.activations import ACT2FN
from einops import rearrange,reduce 

class TsRoPEAttention(nn.Module):
    def __init__(self, layer_idx: int, **cfg):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.get("embed_dim", 768)
        self.num_heads = cfg.get("num_heads", 12)
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = cfg.get("dropout_rate", 0.1)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # 2d RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim//2, max_position_embeddings=cfg.get("max_position_embeddings"))

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        '''
        hidden_states: [bs, seq_len, hidden_size]
        attention_mask: [bs, nvar, num_p]
        '''
        bsz, q_len, _ = hidden_states.size()

        tmp_attn_mask = rearrange(attention_mask, 'b nvar p -> b (nvar p)')
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states) # Bs, L, hidden_size

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        tmp_attn_mask = tmp_attn_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, q_len, q_len).bool()  # bs, 1, L, L

        pos_var, pos_patch = build_2d_position_ids(attention_mask,flatten=True) 
        q_h = query_states[..., : self.head_dim // 2]
        q_w = query_states[..., self.head_dim // 2 :]
        cos_h, sin_h = self.rotary_emb(q_h, seq_len=int(pos_var.max().item()) + 1)
        cos_w, sin_w = self.rotary_emb(q_w, seq_len=int(pos_patch.max().item()) + 1)

        query_states, key_states = apply_rotary_pos_emb_2d(
            query_states, key_states, 
            cos_h, sin_h, 
            cos_w, sin_w, 
            pos_var, pos_patch
        )

        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            tmp_attn_mask, 
            dropout_p=self.attention_dropout
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)


        return attn_output
    
# helper function
def flatten_list(input_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Flatten a nested list of lists into a single list.
    Args:
        input_list (List[List[Tensor]]): Nested list to flatten.
    Returns:
        List[Tensor]: Flattened list.
    """
    return [item for sublist in input_list for item in sublist]

class MultiSizePatchEmbed(nn.Module):
    def __init__(self, base_patch=32, **cfg):
        super().__init__()

        self.base_patch = base_patch  
        hidden_size = cfg['embed_dim']
        intermediate_size = cfg['mlp_ratio'] * hidden_size # 3072
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

        # [ts, time_idx, mask] concatenated together
        self.shared_linear = nn.Linear(base_patch*3, intermediate_size) # putting mask on hidden.
        self.shared_residual = nn.Linear(base_patch*3, hidden_size)

        # MLP embedder ###
        self.dropout = nn.Dropout(cfg['dropout_rate'])
        self.act = ACT2FN['silu']
        self.output_layer = nn.Linear(
            intermediate_size, hidden_size)

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
        
 
    def resize_weight(self, patch_size: int):
        """
        Interpolate weights along the patch dimension to target patch size.
        """

        base_w = self.shared_linear.weight  # [out_dim, base_patch]
        base_b = self.shared_linear.bias

        res_w = self.shared_residual.weight
        res_b = self.shared_residual.bias

        # FlexiViT: interpolate kernel linearly along patch axis
        # interpolate (base_patch, d) -> (patch_size,d)
        new_w = F.interpolate(
            base_w.unsqueeze(1), size=patch_size, mode="linear", align_corners=False
        ).squeeze(1).to(base_w.dtype)

        new_res_w = F.interpolate(
            res_w.unsqueeze(1), size=patch_size, mode="linear", align_corners=False
        ).squeeze(1).to(res_w.dtype)

        return new_w, base_b,new_res_w,res_b


    def forward(self, x_list, attention_mask, time_idx):
        """
        x_list: list of tensors of shape (num_patches, patch_size)
        attention_mask: list of tensors.
        

        Returns:
            list of transformed tensors in the same order.
        """

        amp_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        # group by patch size
        sizes = torch.tensor([x.shape[-1] for x in x_list])
        unique_sizes = sizes.unique(sorted=True)
        N = x_list[0].shape[0] # number of patches
        
        outputs = torch.empty(len(x_list), N, self.intermediate_size, 
                              device=device,dtype=amp_dtype)
        res_outputs = torch.empty(len(x_list), N, self.hidden_size, 
                                  device=device,dtype=amp_dtype)
        
        for psize in unique_sizes.tolist():
            idxs = (sizes == psize).nonzero(as_tuple=True)[0]
            xs = torch.stack([x_list[i] for i in idxs]) # B_g, num_p, ps
            mask = torch.stack([attention_mask[i] for i in idxs]) # B_g, num_p, ps
            ti = torch.stack([time_idx[i] for i in idxs])

            xs = xs.to(device=device, non_blocking=True)
            mask = mask.to(device=device, non_blocking=True)
            ti = ti.to(device=device, non_blocking=True)

            xs = torch.cat([xs,mask,ti],dim=-1) # B_g, num_p, ps*3
            w, b, r_w, r_b = self.resize_weight(psize*3)

            res_outputs[idxs] = F.linear(xs,r_w,r_b)
            outputs[idxs] = F.linear(xs, w, b)

        hid = self.act(outputs) # BS, num_p, intermediate_size
        out = self.dropout(self.output_layer(hid)) # BS, num_p, hidden
        out = out + res_outputs 
        
        return out 
    

class PatchEmbedding(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        patch_size = cfg['patch_size']
        self.patch_size = patch_size

        self.dropout = nn.Dropout(cfg.get('dropout_rate', 0.1))
        hidden_size = cfg['embed_dim']
        self.hidden_layer = nn.Linear(
            patch_size * 3, hidden_size)
        self.act = ACT2FN['silu']
        self.output_layer = nn.Linear(
            hidden_size, hidden_size)
        self.residual_layer = nn.Linear(
            patch_size * 3, hidden_size)
        self.patch_size = patch_size

    def forward(self, x, mask, time_idx):
        '''
        x,mask,time_idx: bs, nvar,L
        '''
        x = rearrange(x, 'bs nvar (nump ps) -> (bs nvar) nump ps', ps=self.patch_size)
        mask = rearrange(mask, 'bs nvar (nump ps) -> (bs nvar) nump ps', ps=self.patch_size)
        time_idx = rearrange(time_idx, 'bs nvar (nump ps) -> (bs nvar) nump ps', ps=self.patch_size)

        x = torch.cat([x, mask,time_idx], dim=-1)
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)
        out = out + res

        return out # bs*nvar, num_p, hidden_size
    
class Attention(nn.Module):
    def __init__(self, layer_idx: int, is_rope=True, **cfg):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_rope = is_rope
        self.hidden_size = cfg.get("embed_dim", 768)
        self.num_heads = cfg.get("num_heads", 12)
        self.sensor_max_len = cfg.get("sensor_max_len", 2880)
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = cfg.get("dropout_rate", 0.1)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    
        if self.is_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, max_position_embeddings=self.sensor_max_len)
        else:
            self.rotary_emb = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None, # index of positions.
            **kwargs,
    ) -> torch.Tensor:
        '''
        hidden_states: [bs, seq_len, hidden_size]
        attention_mask: [bs, 1, seq_len, seq_len]
        position_ids: [bs, seq_len]
        '''

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states) # Bs, L, hidden_size

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.is_rope:
            kv_seq_len = key_states.shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)

        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attention_mask, 
            dropout_p=self.attention_dropout
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)


        return attn_output
    
class CrossAttention(nn.Module):
    def __init__(self, 
                 dim=768, # unifed embed space
                 *,
                 context_dim=384,
                 num_heads=12,
                 dropout_rate=0.1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = int(dim // num_heads) 
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = dropout_rate
        
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(context_dim, dim, bias=True)
        self.v_proj = nn.Linear(context_dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)


    def forward(
            self,
            query,
            context,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        '''
        hidden_states: [bs, seq_len, hidden_size]
        attention_mask: [BS, 1, seq_len, context_len]
        position_ids: [bs, seq_len]
        '''

        bsz, q_len, _ = query.size()
        bsc, k_len, _ = context.size()

        assert bsz == bsc, f"Batch size mismatch: {bsz} vs {bsc}"

        # pre-norm
        query = self.norm(query)
        context = self.context_norm(context)

        query_states = self.q_proj(query)
        key_states = self.k_proj(context)
        value_states = self.v_proj(context) # Bs, L, hidden_size

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)


        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attention_mask, 
            dropout_p=self.attn_dropout
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.dim)
        attn_output = self.o_proj(attn_output) # bs, q_len, dim

        return attn_output
    
class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))



class AllAttention(nn.Module):
    def __init__(self, layer_idx, **cfg):
        super().__init__()
        self.self_attention = TsRoPEAttention(**cfg, layer_idx=layer_idx)
        self.layer_norm = nn.LayerNorm(cfg.get('embed_dim'))
        self.dropout = nn.Dropout(cfg.get('dropout_rate', 0.1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        '''
        ts self attention with residual
        hidden_states: bs (nvar L) d
        attention_mask: bs, nvar, L

        '''

        normed_hidden_states = self.layer_norm(hidden_states) # pre-norm
        attention_output = self.self_attention(
           normed_hidden_states, 
           attention_mask,
        )

        # residual
        hidden_states = hidden_states + self.dropout(attention_output)
        
        return hidden_states

class TimeSelfAttention(nn.Module):
    def __init__(self, layer_idx, **cfg):
        super().__init__()
        self.self_attention = Attention(layer_idx=layer_idx, is_rope=True, **cfg)
        self.layer_norm = nn.LayerNorm(cfg.get('embed_dim', 768))
        self.dropout = nn.Dropout(cfg.get('dropout_rate', 0.1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        '''
        ts self attention with residual
        hidden_states: bs*nvar, L, d
        attention_mask: bs, nvar, L

        '''

        q_len = hidden_states.size(1)
        attention_mask = rearrange(attention_mask, 'b nvar p -> (b nvar) p')  # bs*nvar, L
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, q_len, q_len)  # bs*nvar, 1, L, L
        attention_mask = attention_mask.bool()  # convert to bool

        normed_hidden_states = self.layer_norm(hidden_states) # pre-norm
        attention_output = self.self_attention(
           normed_hidden_states, 
           attention_mask,
           position_ids
        )

        # residual
        hidden_states = hidden_states + self.dropout(attention_output)

        return hidden_states
    
    
class GroupSelfAttention(nn.Module):
    """Self-attention applied along the batch axis masked by the group attention mask"""

    def __init__(self, layer_idx: int, **cfg):
        super().__init__()
        # we don't use RoPE here because there's no natural ordering along the batch axis
        self.self_attention = Attention(layer_idx, is_rope=False, **cfg)
        self.layer_norm = nn.LayerNorm(cfg.get('embed_dim', 768))
        self.dropout = nn.Dropout(cfg.get('dropout_rate', 0.1))

    def _construct_group_mask(self,
                              group_ids: torch.Tensor, 
                              attention_mask: torch.Tensor) -> torch.Tensor:
            
            # construct group_mask (batch, batch) from group ids
            # a cell is True if both row and col had the same group id
            group_mask = group_ids[:, None] == group_ids[None, :]

            # group_mask: bs*nvar, bs*nvar
            # attention_mask: bs*nvar, L
            group_time_mask = torch.einsum("qb, bt -> qbt", group_mask, attention_mask).float() # bs*nvar, bs*nvar, L
            group_time_mask = rearrange(group_time_mask, "q b t -> t 1 q b") # L,1, bs*nvar, bs*nvar
            group_time_mask = group_time_mask.bool()  # convert to bool

            return group_time_mask

    def forward(
            self, 
            hidden_states: torch.Tensor, 
            attention_mask: torch.Tensor,
            group_ids: torch.Tensor,
    ):

        '''
        hidden_states: bs*nvar, L, d
        attention_mask: bs, nvar, L
        group_ids: bs*nvar
        '''

        
        # attention_mask = rearrange(attention_mask, 'b nvar l -> (b nvar) l')  # bs*nvar, L
        # hidden_states = rearrange(hidden_states, 'bs l d -> l bs d',) # L, bs*nvar, d
        # group_attn_mask = self._construct_group_mask(group_ids, attention_mask) #L,1, bs*nvar, bs*nvar 

        BS, nvar, _ = attention_mask.shape
        hidden_states = rearrange(hidden_states, '(bs nvar) l d -> (bs l) nvar d', bs=BS, nvar=nvar)
        attention_mask = rearrange(attention_mask, 'bs nvar l -> (bs l) nvar')  # (bs*L), nvar
        group_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, nvar, nvar).bool()  # (bs*L), 1, nvar, nvar

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.self_attention(
            normed_hidden_states, 
            group_attn_mask,
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        # flip time and batch axes back to their original position
        hidden_states = rearrange(hidden_states, '(bs l) nvar d -> (bs nvar) l d', bs=BS, nvar=nvar)
        # hidden_states = rearrange(hidden_states, "time batch d -> batch time d") # Bs*nvar, L, d


        return hidden_states
    
class AttentionPooling(nn.Module):  
    def __init__(self, 
                 dim=768,
                 mlp_ratio=4, 
                 context_dim=384, 
                 num_heads=12,
                 dropout_rate=0.1):
        super().__init__()

        self.cross_attn = CrossAttention(dim=dim,
                                         context_dim=context_dim,
                                         num_heads=num_heads,
                                         dropout_rate=dropout_rate)
        
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_layer = MLP(
            hidden_size=dim,
            intermediate_size=dim * mlp_ratio,
            hidden_act='silu',
        )

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, context, attn_mask=None):
        # x: BS, num_query, dim
        # context: BS, num_kv, context_dim
        # attn_mask: BS, nvar, num_p,
        b,n,_ = x.shape
        kv_len = context.shape[1]
        
        attn_mask = rearrange(attn_mask, 'b nvar p -> b (nvar p)')
        attn_mask = attn_mask.view(b, 1, 1, kv_len).expand(b, 1, n, kv_len).bool() 

        x = self.cross_attn(x, context, attn_mask)
        x = x + self.ffn_layer(self.ffn_norm(x))
        x = self.post_norm(x)

        return x


class SensorEncoderLayer(nn.Module):
    def __init__(self, layer_idx: int, **cfg):
        super().__init__()

        hidden_size = cfg['embed_dim']
        intermediate_size = cfg['mlp_ratio'] * hidden_size

        self.channel_attn_type = cfg.get('channel_attn_type', 'group_attn')
        if self.channel_attn_type == 'group_attn':
            self.ts_attn = TimeSelfAttention(layer_idx=layer_idx, **cfg) # pre-norm
            self.group_attn = GroupSelfAttention(layer_idx=layer_idx, **cfg) # pre-norm
        elif self.channel_attn_type == 'univariate':
            self.ts_attn = TimeSelfAttention(layer_idx=layer_idx, **cfg)
        else:
            self.ts_attn = AllAttention(layer_idx=layer_idx, **cfg) 

        self.norm = nn.LayerNorm(hidden_size) # post-norm

        self.ffn_layer = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act='silu',
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            group_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:

        
        if self.channel_attn_type == 'group_attn':
            '''
            Time self attention with residual
            hidden_states: bs*nvar, L, d
            attention_mask: bs, nvar, L
            group_attention_mask: bs*nvar, bs*nvar
            '''
            hidden_states = self.ts_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            ) # handled residual
           
            
            hidden_states = self.group_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                group_ids=group_ids,
            ) # handled residual

            # Fully Connected
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            hidden_states = self.ffn_layer(hidden_states)
            hidden_states = residual + hidden_states
        
        elif self.channel_attn_type == 'univariate':
            # hidden_states: bs*nvar, L, d
            hidden_states = self.ts_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            ) # handled residual

            # Fully Connected
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            hidden_states = self.ffn_layer(hidden_states)
            hidden_states = residual + hidden_states

        else:
            # hidden_states: bs (nvar L) d
            hidden_states = self.ts_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            ) # b (nvar l) d

            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            hidden_states = self.ffn_layer(hidden_states)
            hidden_states = residual + hidden_states

        
        return hidden_states
    

    
class SensorTransformerModel(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        patch_size = cfg.get('patch_size', None)
        self.patch_size = patch_size
        if patch_size is not None:
            # fixed patch size embedder
            self.patch_embed = PatchEmbedding(**cfg)
        else:
            self.patch_embed = MultiSizePatchEmbed(**cfg)

        self.blocks = nn.ModuleList(
            [SensorEncoderLayer(layer_idx, **cfg)
             for layer_idx in range(cfg['depth'])]
        )
        self.norm = torch.nn.LayerNorm(cfg['embed_dim'])
        self.embed_dim = cfg['embed_dim']
        self.channel_attn_type = cfg.get('channel_attn_type', 'group_attn') # group_attn, all_attn, univariate
    
    def forward(
            self,
            input_ids,
            attention_mask,
            time_index,):
        
        
        if self.patch_size is None:
            '''
            input_ids: list of list of tensor # BS, nvar, num_p, patch_size
            attention_mask: same as input_ids

            self.patch_embed will handle device.
            '''
            BS = len(input_ids)
            flat_input_ids = flatten_list(input_ids)
            flat_attention_mask = flatten_list(attention_mask)
            flat_time_index = flatten_list(time_index)

            # embed each variable separately
            hidden_states = self.patch_embed(flat_input_ids,flat_attention_mask,flat_time_index)  # (bs*nvar, seq_len, embed_dim)

            attention_mask = self._get_self_attn_mask(attention_mask).to(hidden_states.device)  # BS, nvar, num_p
            position_ids = self._build_rope_position_ids(attention_mask)  # BS, nvar, num_p
            position_ids = rearrange(position_ids, 'b nvar p -> (b nvar) p')  # BS*nvar, num_p

        else:
            '''
            input_ids: tensor # BS, nvar, L
            attention_mask: tensor # BS, nvar, L
            time_index: tensor # BS, nvar, L
            '''
          
            BS, nvar, L = input_ids.shape
            hidden_states = self.patch_embed(input_ids, attention_mask, time_index)  # (bs*nvar, seq_len, embed_dim)
            # transform pixel-level attn mask (BS, nvar, L)to patch-level attn mask (BS, nvar, num_p), element would be 1 if all pixel is 1,if all pixel is 0, then is 0
            attention_mask = reduce(
                attention_mask,
                'b v (p ps) -> b v p',
                'max',
                ps=self.patch_size
            )
             
            position_ids = self._build_rope_position_ids(attention_mask)  # BS, nvar, num_p
            position_ids = rearrange(position_ids, 'b nvar p -> (b nvar) p')  # BS*nvar, num_p

        if self.channel_attn_type == 'all_attn':
            hidden_states = rearrange(hidden_states, '(b nvar) l d -> b (nvar l) d', b=BS)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                attention_mask=attention_mask,
                group_ids=None, # legacy argument
                position_ids=position_ids,
            ) # bs*nvar, seq, emb or bs (nvar l) d

        if self.channel_attn_type == 'group_attn':
            hidden_states = rearrange(hidden_states, '(b nvar) l d -> b (nvar l) d', b=BS)

        hidden_states = self.norm(hidden_states) # (Bs*nvar), seq, emb

        return hidden_states, attention_mask
    
    def _build_rope_position_ids(self,attention_mask):
        """
        attention_mask: Tensor [BS, nvar, num_p]
        returns: LongTensor [BS, nvar, num_p]
        """
        assert attention_mask.dim() == 3
        BS, nvar, num_p = attention_mask.shape

        mask = attention_mask.to(torch.long)

        # position index increases inside each variable
        pos = (mask.cumsum(dim=-1) - 1) * mask         # [BS, nvar, num_p]

        return pos

    def _get_self_attn_mask(self,attn_mask_list):
        """
        Collapse a nested list of attention masks from shape
            [BS][nvar][num_p, patch_size]
        into tensors of shape [BS, nvar, num_p].

        Args:
            attention_mask (list[list[Tensor]]):
                Each tensor has shape [num_p, patch_size], and all have the same shape.

        Returns:
            torch.Tensor (BS, nvar, num_p)
        """
        collapsed_batch = []
        for sample_masks in attn_mask_list:  # loop over batch
            # collapse each [num_p, patch_size] → [num_p]
            nvar_collapsed = [
                (var_mask.sum(dim=-1) > 0).to(var_mask.dtype) for var_mask in sample_masks
            ]
            nvar_collapsed = torch.stack(nvar_collapsed, dim=0)  # [nvar, num_p]
            collapsed_batch.append(nvar_collapsed)

        collapsed_batch = torch.stack(collapsed_batch, dim=0)  # [BS, nvar, num_p]
        return collapsed_batch

    def _get_group_ids(self,attn_mask_list):
        """
        attn_mask_list: list of list of tensor
            BS, nvar
            each tensor is shape (num_p, patch_size)
        
        Returns:
            group_mask: (BS*nvar, BS*nvar) boolean tensor
                True means same group
                False means different group
        """
        BS = len(attn_mask_list)
        nvar = len(attn_mask_list[0])

        # build group ids
        # each sample i repeats nvar times
        group_ids = torch.arange(BS).repeat_interleave(nvar)  # (BS*nvar)

        return group_ids




if __name__ == "__main__":
    from model_factory.coca import Config
    cfg = Config(embed_dim=384,
                 num_heads=6,
                 mlp_ratio=4,
                 depth=12,
                 dropout_rate=0.1,)
    sensor_model = SensorTransformerModel(**cfg)
    dummy_input = [[torch.randn(14,40),torch.randn(14,40)],[torch.randn(14,40),torch.randn(14,30)]]
    mask = [[torch.ones(14,40),torch.zeros(14,40)],[torch.zeros(14,40),torch.zeros(14,30)]]
    time_idx = [[torch.ones(14,40),torch.ones(14,40)],[torch.ones(14,40),torch.ones(14,30)]]

    out, attn_mask = sensor_model(dummy_input,mask,time_idx)
    print(out.shape)  # expect (2*2, max_num_patches, embed
    # python -m model_factory.ts_transformer