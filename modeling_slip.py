"""
SLIP: Sensor Language Integrated Pre-training
Self-contained model file for HuggingFace Hub (trust_remote_code=True).

Usage:
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained("LeoChen085/SLIP", trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("LeoChen085/SLIP", trust_remote_code=True)

    # Task-specific checkpoint (download manually):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    state_dict = load_file(hf_hub_download("LeoChen085/SLIP", "har.safetensors"))
    model.load_state_dict(state_dict, strict=False)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from einops import rearrange, repeat, reduce
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.activations import ACT2FN
from configuration_slip import SLIPConfig


# ═══════════════════════════════════════════════════════════════
# Positional Embeddings
# ═══════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=10000, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim,
                          2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_2d(q, k, cos_h, sin_h, cos_w, sin_w, pos_h, pos_w, unsqueeze_dim=1):
    Dh = q.shape[-1]
    q_h, q_w = q.split(Dh // 2, dim=-1)
    k_h, k_w = k.split(Dh // 2, dim=-1)
    q_h, k_h = apply_rotary_pos_emb(q_h, k_h, cos_h, sin_h, pos_h.long(), unsqueeze_dim=unsqueeze_dim)
    q_w, k_w = apply_rotary_pos_emb(q_w, k_w, cos_w, sin_w, pos_w.long(), unsqueeze_dim=unsqueeze_dim)
    return torch.cat([q_h, q_w], dim=-1), torch.cat([k_h, k_w], dim=-1)


def build_2d_position_ids(attention_mask, flatten=True):
    B, V, P = attention_mask.shape
    mask = attention_mask.to(dtype=torch.long)
    pos_patch = (mask.cumsum(dim=-1) - 1) * mask
    var_valid = mask.any(dim=-1).to(dtype=torch.long)
    pos_var_base = (var_valid.cumsum(dim=1) - 1) * var_valid
    pos_var = pos_var_base.unsqueeze(-1).expand(B, V, P) * mask
    if flatten:
        return pos_var.reshape(B, V * P).long(), pos_patch.reshape(B, V * P).long()
    return pos_var.long(), pos_patch.long()


# ═══════════════════════════════════════════════════════════════
# Sensor Encoder Components
# ═══════════════════════════════════════════════════════════════

def flatten_list(input_list):
    return [item for sublist in input_list for item in sublist]


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class TsRoPEAttention(nn.Module):
    def __init__(self, layer_idx, **cfg):
        super().__init__()
        self.hidden_size = cfg.get("embed_dim", 768)
        self.num_heads = cfg.get("num_heads", 12)
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = cfg.get("dropout_rate", 0.1)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim // 2, max_position_embeddings=cfg.get("max_position_embeddings"))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        tmp_attn_mask = rearrange(attention_mask, 'b nvar p -> b (nvar p)')
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        tmp_attn_mask = tmp_attn_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, q_len, q_len).bool()
        pos_var, pos_patch = build_2d_position_ids(attention_mask, flatten=True)
        cos_h, sin_h = self.rotary_emb(query_states, seq_len=int(pos_var.max().item()) + 1)
        cos_w, sin_w = self.rotary_emb(query_states, seq_len=int(pos_patch.max().item()) + 1)
        query_states, key_states = apply_rotary_pos_emb_2d(
            query_states, key_states, cos_h, sin_h, cos_w, sin_w, pos_var, pos_patch)
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, tmp_attn_mask, dropout_p=self.attention_dropout)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class MultiSizePatchEmbed(nn.Module):
    def __init__(self, base_patch=32, **cfg):
        super().__init__()
        self.base_patch = base_patch
        hidden_size = cfg['embed_dim']
        intermediate_size = cfg['mlp_ratio'] * hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.shared_linear = nn.Linear(base_patch * 3, intermediate_size)
        self.shared_residual = nn.Linear(base_patch * 3, hidden_size)
        self.dropout = nn.Dropout(cfg['dropout_rate'])
        self.act = ACT2FN['silu']
        self.output_layer = nn.Linear(intermediate_size, hidden_size)

    def resize_weight(self, patch_size):
        base_w, base_b = self.shared_linear.weight, self.shared_linear.bias
        res_w, res_b = self.shared_residual.weight, self.shared_residual.bias
        new_w = F.interpolate(base_w.unsqueeze(1), size=patch_size, mode="linear", align_corners=False).squeeze(1).to(base_w.dtype)
        new_res_w = F.interpolate(res_w.unsqueeze(1), size=patch_size, mode="linear", align_corners=False).squeeze(1).to(res_w.dtype)
        return new_w, base_b, new_res_w, res_b

    def forward(self, x_list, attention_mask, time_idx):
        device = self.shared_linear.weight.device
        dtype = self.shared_linear.weight.dtype
        sizes = torch.tensor([x.shape[-1] for x in x_list])
        unique_sizes = sizes.unique(sorted=True)
        N = x_list[0].shape[0]
        outputs = torch.empty(len(x_list), N, self.intermediate_size, device=device, dtype=dtype)
        res_outputs = torch.empty(len(x_list), N, self.hidden_size, device=device, dtype=dtype)
        for psize in unique_sizes.tolist():
            idxs = (sizes == psize).nonzero(as_tuple=True)[0]
            xs = torch.stack([x_list[i] for i in idxs]).to(device=device, non_blocking=True)
            mask = torch.stack([attention_mask[i] for i in idxs]).to(device=device, non_blocking=True)
            ti = torch.stack([time_idx[i] for i in idxs]).to(device=device, non_blocking=True)
            xs = torch.cat([xs, mask, ti], dim=-1)
            w, b, r_w, r_b = self.resize_weight(psize * 3)
            res_outputs[idxs] = F.linear(xs, r_w, r_b)
            outputs[idxs] = F.linear(xs, w, b)
        return self.dropout(self.output_layer(self.act(outputs))) + res_outputs


class PatchEmbedding(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        patch_size = cfg['patch_size']
        self.patch_size = patch_size
        self.dropout = nn.Dropout(cfg.get('dropout_rate', 0.1))
        hidden_size = cfg['embed_dim']
        self.hidden_layer = nn.Linear(patch_size * 3, hidden_size)
        self.act = ACT2FN['silu']
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.residual_layer = nn.Linear(patch_size * 3, hidden_size)

    def forward(self, x, mask, time_idx):
        x = rearrange(x, 'bs nvar (nump ps) -> (bs nvar) nump ps', ps=self.patch_size)
        mask = rearrange(mask, 'bs nvar (nump ps) -> (bs nvar) nump ps', ps=self.patch_size)
        time_idx = rearrange(time_idx, 'bs nvar (nump ps) -> (bs nvar) nump ps', ps=self.patch_size)
        x = torch.cat([x, mask, time_idx], dim=-1)
        return self.dropout(self.output_layer(self.act(self.hidden_layer(x)))) + self.residual_layer(x)


class Attention(nn.Module):
    def __init__(self, layer_idx, is_rope=True, **cfg):
        super().__init__()
        self.is_rope = is_rope
        self.hidden_size = cfg.get("embed_dim", 768)
        self.num_heads = cfg.get("num_heads", 12)
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = cfg.get("dropout_rate", 0.1)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.is_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=cfg.get("sensor_max_len", 2880))

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.is_rope:
            cos, sin = self.rotary_emb(value_states, seq_len=key_states.shape[-2])
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attention_mask, dropout_p=self.attention_dropout)
        return self.o_proj(attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size))


class CrossAttention(nn.Module):
    def __init__(self, dim=768, *, context_dim=384, num_heads=12, dropout_rate=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = dropout_rate
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(context_dim, dim, bias=True)
        self.v_proj = nn.Linear(context_dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, query, context, attention_mask=None, **kwargs):
        bsz, q_len, _ = query.size()
        assert context.size(0) == bsz, (
            f"Context batch size ({context.size(0)}) must match query batch size ({bsz}). "
            f"Ensure sensor and text inputs have the same batch size."
        )
        k_len = context.size(1)
        query = self.norm(query)
        context = self.context_norm(context)
        q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=self.attn_dropout)
        return self.o_proj(attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.dim))


class AllAttention(nn.Module):
    def __init__(self, layer_idx, **cfg):
        super().__init__()
        self.self_attention = TsRoPEAttention(layer_idx=layer_idx, **cfg)
        self.layer_norm = nn.LayerNorm(cfg.get('embed_dim'))
        self.dropout = nn.Dropout(cfg.get('dropout_rate', 0.1))

    def forward(self, hidden_states, attention_mask):
        return hidden_states + self.dropout(self.self_attention(self.layer_norm(hidden_states), attention_mask))


class TimeSelfAttention(nn.Module):
    def __init__(self, layer_idx, **cfg):
        super().__init__()
        self.self_attention = Attention(layer_idx=layer_idx, is_rope=True, **cfg)
        self.layer_norm = nn.LayerNorm(cfg.get('embed_dim', 768))
        self.dropout = nn.Dropout(cfg.get('dropout_rate', 0.1))

    def forward(self, hidden_states, attention_mask, position_ids):
        q_len = hidden_states.size(1)
        am = rearrange(attention_mask, 'b nvar p -> (b nvar) p')
        am = am.unsqueeze(1).unsqueeze(2).expand(-1, 1, q_len, q_len).bool()
        return hidden_states + self.dropout(self.self_attention(self.layer_norm(hidden_states), am, position_ids))


class GroupSelfAttention(nn.Module):
    def __init__(self, layer_idx, **cfg):
        super().__init__()
        self.self_attention = Attention(layer_idx, is_rope=False, **cfg)
        self.layer_norm = nn.LayerNorm(cfg.get('embed_dim', 768))
        self.dropout = nn.Dropout(cfg.get('dropout_rate', 0.1))

    def forward(self, hidden_states, attention_mask, group_ids):
        BS, nvar, _ = attention_mask.shape
        hidden_states = rearrange(hidden_states, '(bs nvar) l d -> (bs l) nvar d', bs=BS, nvar=nvar)
        am = rearrange(attention_mask, 'bs nvar l -> (bs l) nvar')
        group_attn_mask = am.unsqueeze(1).unsqueeze(2).expand(-1, 1, nvar, nvar).bool()
        hidden_states = hidden_states + self.dropout(self.self_attention(self.layer_norm(hidden_states), group_attn_mask))
        return rearrange(hidden_states, '(bs l) nvar d -> (bs nvar) l d', bs=BS, nvar=nvar)


class AttentionPooling(nn.Module):
    def __init__(self, dim=768, mlp_ratio=4, context_dim=384, num_heads=12, dropout_rate=0.1):
        super().__init__()
        self.cross_attn = CrossAttention(dim=dim, context_dim=context_dim, num_heads=num_heads, dropout_rate=dropout_rate)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_layer = MLP(hidden_size=dim, intermediate_size=dim * mlp_ratio, hidden_act='silu')
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, context, attn_mask=None):
        b, n, _ = x.shape
        kv_len = context.shape[1]
        attn_mask = rearrange(attn_mask, 'b nvar p -> b (nvar p)')
        attn_mask = attn_mask.view(b, 1, 1, kv_len).expand(b, 1, n, kv_len).bool()
        x = self.cross_attn(x, context, attn_mask)
        x = x + self.ffn_layer(self.ffn_norm(x))
        return self.post_norm(x)


class SensorEncoderLayer(nn.Module):
    def __init__(self, layer_idx, **cfg):
        super().__init__()
        hidden_size = cfg['embed_dim']
        self.channel_attn_type = cfg.get('channel_attn_type', 'group_attn')
        if self.channel_attn_type == 'group_attn':
            self.ts_attn = TimeSelfAttention(layer_idx=layer_idx, **cfg)
            self.group_attn = GroupSelfAttention(layer_idx=layer_idx, **cfg)
        elif self.channel_attn_type == 'univariate':
            self.ts_attn = TimeSelfAttention(layer_idx=layer_idx, **cfg)
        else:
            self.ts_attn = AllAttention(layer_idx=layer_idx, **cfg)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn_layer = MLP(hidden_size=hidden_size, intermediate_size=cfg['mlp_ratio'] * hidden_size, hidden_act='silu')

    def forward(self, hidden_states, attention_mask=None, group_ids=None, position_ids=None):
        if self.channel_attn_type == 'group_attn':
            hidden_states = self.ts_attn(hidden_states, attention_mask, position_ids)
            hidden_states = self.group_attn(hidden_states, attention_mask, group_ids)
        elif self.channel_attn_type == 'univariate':
            hidden_states = self.ts_attn(hidden_states, attention_mask, position_ids)
        else:
            hidden_states = self.ts_attn(hidden_states, attention_mask)
        residual = hidden_states
        return residual + self.ffn_layer(self.norm(hidden_states))


class SensorTransformerModel(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        patch_size = cfg.get('patch_size', None)
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(**cfg) if patch_size else MultiSizePatchEmbed(**cfg)
        self.blocks = nn.ModuleList([SensorEncoderLayer(i, **cfg) for i in range(cfg['depth'])])
        self.norm = nn.LayerNorm(cfg['embed_dim'])
        self.embed_dim = cfg['embed_dim']
        self.channel_attn_type = cfg.get('channel_attn_type', 'group_attn')

    def forward(self, input_ids, attention_mask, time_index):
        if self.patch_size is None:
            BS = len(input_ids)
            hidden_states = self.patch_embed(flatten_list(input_ids), flatten_list(attention_mask), flatten_list(time_index))
            attention_mask = self._get_self_attn_mask(attention_mask).to(hidden_states.device)
            position_ids = rearrange(self._build_rope_position_ids(attention_mask), 'b nvar p -> (b nvar) p')
        else:
            BS = input_ids.shape[0]
            hidden_states = self.patch_embed(input_ids, attention_mask, time_index)
            attention_mask = reduce(attention_mask, 'b v (p ps) -> b v p', 'max', ps=self.patch_size)
            position_ids = rearrange(self._build_rope_position_ids(attention_mask), 'b nvar p -> (b nvar) p')

        if self.channel_attn_type == 'all_attn':
            hidden_states = rearrange(hidden_states, '(b nvar) l d -> b (nvar l) d', b=BS)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, attention_mask=attention_mask, group_ids=None, position_ids=position_ids)
        if self.channel_attn_type == 'group_attn':
            hidden_states = rearrange(hidden_states, '(b nvar) l d -> b (nvar l) d', b=BS)
        return self.norm(hidden_states), attention_mask

    def _build_rope_position_ids(self, attention_mask):
        mask = attention_mask.to(torch.long)
        return (mask.cumsum(dim=-1) - 1) * mask

    def _get_self_attn_mask(self, attn_mask_list):
        collapsed = []
        for sample_masks in attn_mask_list:
            collapsed.append(torch.stack([(m.sum(dim=-1) > 0).to(m.dtype) for m in sample_masks], dim=0))
        return torch.stack(collapsed, dim=0)


# ═══════════════════════════════════════════════════════════════
# Multimodal Gemma
# ═══════════════════════════════════════════════════════════════

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Gemma3MultimodalLayer(nn.Module):
    def __init__(self, original_layer, cross_attn_block):
        super().__init__()
        self.original_layer = original_layer
        self.cross_attn_block = cross_attn_block
        self.vis_x = None

    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def forward(self, hidden_states, **kwargs):
        assert self.vis_x is not None, "vis_x must be set before forward pass."
        outputs = self.original_layer(hidden_states, **kwargs)
        hidden_states = self.cross_attn_block(outputs[0], context=self.vis_x)
        return (hidden_states,) + outputs[1:]


class Gemma3MultimodalModel(nn.Module):
    def __init__(self, model_id="google/gemma-3-270m", init_from_pretrained=False, split_layer=12, dtype=None):
        super().__init__()
        if init_from_pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            config.torch_dtype = dtype or torch.float32
            self.model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True)

        self.split_layer = split_layer
        hidden_size = self.model.config.hidden_size
        num_heads = self.model.config.num_attention_heads
        self.hidden_size = hidden_size

        for i in range(split_layer, len(self.model.model.layers)):
            cross_attn = CrossAttention(
                dim=hidden_size, context_dim=hidden_size, num_heads=num_heads, dropout_rate=0.1)
            self.model.model.layers[i] = Gemma3MultimodalLayer(
                self.model.model.layers[i], Residual(cross_attn))

    def condition_image(self, image_embeds):
        self.image_embeds = image_embeds
        for layer in self.model.model.layers:
            if isinstance(layer, Gemma3MultimodalLayer):
                layer.condition_vis_x(self.image_embeds)

    def forward(self, input_ids, attention_mask=None, return_embeddings=False, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        text_sentence_embedding = outputs.hidden_states[self.split_layer][:, -1, :]
        if return_embeddings:
            return outputs
        return text_sentence_embedding, outputs.logits


# ═══════════════════════════════════════════════════════════════
# SLIP Model (PreTrainedModel for HuggingFace Auto* classes)
# ═══════════════════════════════════════════════════════════════

def masked_mean(t, mask, dim=1, eps=1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim=dim)
    denom = mask.sum(dim=dim).clamp(min=eps)
    return numer / denom


class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        return F.normalize(self.to_latents(x), dim=-1)


class SLIPPreTrainedModel(PreTrainedModel):
    config_class = SLIPConfig
    base_model_prefix = "slip"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class SLIPModel(SLIPPreTrainedModel):
    """
    SLIP: Sensor Language Integrated Pre-training.

    Usage:
        model = AutoModel.from_pretrained("LeoChen085/SLIP", trust_remote_code=True)
    """

    def __init__(self, config: SLIPConfig):
        super().__init__(config)

        # Sensor encoder
        sensor_cfg = config.sensor_encoder
        self.sensor_encoder = SensorTransformerModel(**sensor_cfg)
        dim = self.sensor_encoder.embed_dim

        # Multimodal LLM (init from scratch — weights come from safetensors)
        self.multimodalModel = Gemma3MultimodalModel(
            config.llm_model_name,
            init_from_pretrained=False,
            split_layer=config.split_layer,
            dtype=getattr(config, "torch_dtype", None),
        )

        lm_dim = self.multimodalModel.hidden_size
        common_dim = config.common_dim

        # Attention pooling
        num_img_queries = config.num_img_queries
        if num_img_queries > 0:
            self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, common_dim))
            self.img_attn_pool = AttentionPooling(
                dim=common_dim, context_dim=dim, num_heads=config.num_heads)
            dim = common_dim

        # Bridge projections
        self.img_to_latents = EmbedToLatents(dim, common_dim)
        self.text_to_latents = EmbedToLatents(common_dim, common_dim)

        # Temperature
        self.temperature = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.temperature_max = math.log(1 / 0.07)

    def embed_sensor(self, sensors, sensor_attn_mask=None, time_index=None):
        sensor_tokens, attn_mask = self.sensor_encoder(sensors, sensor_attn_mask, time_index=time_index)
        if hasattr(self, "img_attn_pool"):
            img_queries = repeat(self.img_queries, "n d -> b n d", b=sensor_tokens.shape[0])
            sensor_tokens = self.img_attn_pool(img_queries, sensor_tokens, attn_mask)
        return sensor_tokens, attn_mask.bool()

    def forward(self, text=None, sensors=None, **kwargs):
        """
        Forward pass for contrastive + captioning training.
        For inference, use get_embedding(), get_sensor_embedding(), or generate().
        """
        sensor_hidden, sensor_mask = self.embed_sensor(
            sensors=sensors["input_ids"], sensor_attn_mask=sensors["attention_mask"],
            time_index=sensors["time_index"])
        self.multimodalModel.condition_image(sensor_hidden)
        text_hidden, logits = self.multimodalModel(
            input_ids=text["input_ids"][:, :-1], attention_mask=text["attention_mask"][:, :-1])
        text_hidden = self.text_to_latents(text_hidden)
        sensor_hidden = self.img_to_latents(sensor_hidden)
        return {"text_hidden": text_hidden, "sensor_hidden": sensor_hidden, "logits": logits}

    @torch.no_grad()
    def get_embedding(self, text, sensors):
        sensor_hidden, sensor_mask = self.embed_sensor(
            sensors=sensors["input_ids"], sensor_attn_mask=sensors["attention_mask"],
            time_index=sensors["time_index"])
        self.multimodalModel.condition_image(sensor_hidden)
        text_hidden, _ = self.multimodalModel(
            input_ids=text["input_ids"][:, :-1], attention_mask=text["attention_mask"][:, :-1])
        text_hidden = self.text_to_latents(text_hidden)
        sensor_hidden = self.img_to_latents(sensor_hidden)
        if hasattr(self, "img_attn_pool"):
            sensor_hidden = sensor_hidden[:, 0, :]
        else:
            sensor_hidden = masked_mean(sensor_hidden, rearrange(sensor_mask, "b n p -> b (n p) 1"), dim=1)
        return text_hidden, sensor_hidden

    @torch.no_grad()
    def get_sensor_embedding(self, input_ids, mask, time_index):
        sensor_hidden, sensor_mask = self.embed_sensor(sensors=input_ids, sensor_attn_mask=mask, time_index=time_index)
        sensor_hidden = self.img_to_latents(sensor_hidden)
        if hasattr(self, "img_attn_pool"):
            sensor_hidden = sensor_hidden[:, 0, :]
        else:
            sensor_hidden = masked_mean(sensor_hidden, rearrange(sensor_mask, "b n p -> b (n p) 1"), dim=1)
        return sensor_hidden

    @torch.no_grad()
    def generate(self, text, sensors, **generate_kwargs):
        sensor_hidden, _ = self.embed_sensor(
            sensors=sensors["input_ids"], sensor_attn_mask=sensors["attention_mask"],
            time_index=sensors["time_index"])
        self.multimodalModel.condition_image(sensor_hidden)
        return self.multimodalModel.model.generate(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"],
            max_new_tokens=generate_kwargs.get("max_new_tokens", 300),
            do_sample=generate_kwargs.get("do_sample", False),
            num_beams=generate_kwargs.get("num_beams", 1))

    def sft_training(self, text, sensors, return_output=False):
        sensor_hidden, _ = self.embed_sensor(
            sensors=sensors["input_ids"], sensor_attn_mask=sensors["attention_mask"],
            time_index=sensors["time_index"])
        self.multimodalModel.condition_image(sensor_hidden)
        outputs = self.multimodalModel.model(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"], return_dict=True)
        if return_output:
            return outputs
        logits = outputs.logits
        labels = text["labels"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
            reduction="none", ignore_index=-100)
        if "loss_weights" in text:
            loss_weights = text["loss_weights"][:, 1:].contiguous().view(-1)
            loss = (ce * loss_weights).sum() / loss_weights.sum()
        else:
            loss = ce.mean()
        return {"loss": loss}
