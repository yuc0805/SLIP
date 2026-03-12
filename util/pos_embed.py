import numpy as np
import torch


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    Create 1D sine-cosine positional embeddings.

    Args:
        embed_dim (int): Dimension of the embedding (must be even)
        length (int): Number of positions (sequence length)
        cls_token (bool): Whether to include an extra zero vector for [CLS] token

    Returns:
        np.ndarray of shape (length, embed_dim) or (1+length, embed_dim) if cls_token=True
    """
    # position indices 0 ... length-1
    pos = np.arange(length, dtype=np.float32)

    # get embedding from grid
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)  # (L, D)

    # optionally add CLS token embedding
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):  
    # --------------------------------------------------------
    # 2D sine-cosine position embedding
    # References:
    # Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
    # MoCo v3: https://github.com/facebookresearch/moco-v3
    # --------------------------------------------------------

    grid_h = np.arange(grid_size[0], dtype=np.float32) 
    grid_w = np.arange(grid_size[1], dtype=np.float32) 
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):  
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  #changed(H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  #changed (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position 
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def interpolate_pos_embed(model, checkpoint_model, orig_size, new_size):
    '''
    Input: model: the class is definging for downstream
           checkpoint_model: pre-train weight
           orig_size = patch size in the ckpt
           new_size = patch size in the current model
    '''

    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed'] # 1 x 560 x 768 (1 x num_patches x E)
        embedding_size = pos_embed_checkpoint.shape[-1] # 768

        # number of special tokens (e.g. in this case num_extra_tokens = 1 for the cls token)
        num_patches = model.patch_embed.num_patches  
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches 
        
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] # old positions
            pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed



# RoPE: https://huggingface.co/thuml/sundial-base-128m/blob/main/modeling_sundial.py
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=10000, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim,
                          2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device,
                         dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# two dimensional version
def apply_rotary_pos_emb_2d(q, k, 
                            cos_h, sin_h, 
                            cos_w, sin_w, 
                            pos_h, pos_w, 
                            unsqueeze_dim=1):
    """
    q, k: [B, heads, N, Dh]
    cos_h, sin_h: caches from 1D rotary with dim = Dh // 2 for the first axis
    cos_w, sin_w: caches from 1D rotary with dim = Dh // 2 for the second axis
    pos_h, pos_w: [B, N] integer positions for each token along the two axes
    returns q_out, k_out with same shape as q, k
    """
    Dh = q.shape[-1]
    assert Dh % 4 == 0, "head dim must be divisible by 4 so each half is even for rotate_half"

    # split channel dim into two halves
    q_h, q_w = q.split(Dh // 2, dim=-1)
    k_h, k_w = k.split(Dh // 2, dim=-1)

    # apply 1D RoPE on each half with its own positions
    pos_h = pos_h.long()
    pos_w = pos_w.long()
    q_h, k_h = apply_rotary_pos_emb(q_h, k_h, cos_h, sin_h, pos_h, unsqueeze_dim=unsqueeze_dim)
    q_w, k_w = apply_rotary_pos_emb(q_w, k_w, cos_w, sin_w, pos_w, unsqueeze_dim=unsqueeze_dim)

    # concat back
    q_out = torch.cat([q_h, q_w], dim=-1)
    k_out = torch.cat([k_h, k_w], dim=-1)
    return q_out, k_out


def build_2d_position_ids(attention_mask: torch.Tensor, 
                          flatten: bool = True):
    """
    attention_mask: Tensor [BS, nvar, num_p] with 1 for valid patches, 0 for padding.

    Returns:
        If flatten is True:
            pos_var_flat: LongTensor [BS, nvar*num_p]
            pos_patch_flat: LongTensor [BS, nvar*num_p]
        Else:
            pos_var:  LongTensor [BS, nvar, num_p]
            pos_patch: LongTensor [BS, nvar, num_p]
    """
    assert attention_mask.dim() == 3, "attention_mask must be [BS, nvar, num_p]"
    B, V, P = attention_mask.shape
    mask = attention_mask.to(dtype=torch.long)

    # per patch index within each variable, ignores padding
    pos_patch = (mask.cumsum(dim=-1) - 1) * mask                      # [B, V, P]

    # per variable index, ignores variables that are entirely padded
    var_valid = mask.any(dim=-1).to(dtype=torch.long)                 # [B, V]
    pos_var_base = (var_valid.cumsum(dim=1) - 1) * var_valid          # [B, V]
    pos_var = pos_var_base.unsqueeze(-1).expand(B, V, P) * mask       # [B, V, P]

    if flatten:
        return pos_var.reshape(B, V * P).long(), pos_patch.reshape(B, V * P).long()
    
    return pos_var.long(), pos_patch.long()

def build_1d_position_ids(attention_mask: torch.Tensor):
    """
    Build 1D position ids for [BS, nvar, num_p],
    output shape [BS * nvar, num_p].

    Each (batch, variable) pair gets its own 1D position index sequence
    along the patch axis, skipping padded positions.

    Args:
        attention_mask: Tensor [BS, nvar, num_p], 1 for valid, 0 for padding.

    Returns:
        pos_ids: LongTensor [BS * nvar, num_p]
    """
    assert attention_mask.dim() == 3, "attention_mask must be [BS, nvar, num_p]"
    B, V, P = attention_mask.shape
    mask = attention_mask.to(dtype=torch.long)

    # Compute per-variable cumulative index
    pos_ids = (mask.cumsum(dim=-1) - 1) * mask  # [B, V, P]

    # Reshape to [BS * nvar, num_p]
    pos_ids = pos_ids.view(B * V, P).long()
    
    return pos_ids