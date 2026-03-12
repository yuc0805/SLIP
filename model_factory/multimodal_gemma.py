import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoConfig, AutoModelForCausalLM
from model_factory.ts_transformer import CrossAttention

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
        """Forward all unknown attributes to the original layer."""
        # This is CRITICAL for 'attention_type' and other internal HF flags
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def forward(self, hidden_states, **kwargs):
        # 1. Run the original unimodal Gemma Layer (Self-Attn + MLP)
        # have to have self.vis_x
        assert self.vis_x is not None, "vis_x must be set before forward pass."
        
        outputs = self.original_layer(hidden_states, **kwargs) # gemma layer output
        hidden_states = outputs[0]
        hidden_states = self.cross_attn_block(hidden_states, context=self.vis_x)
        
        return (hidden_states,) + outputs[1:] # make hf happy


class Gemma3MultimodalModel(nn.Module):
    def __init__(self, 
                 model_id="google/gemma-3-270m",
                 post_train = True, 
                 split_layer=12):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        
        if post_train:
            # Load pre-trained weights
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
        else:
            # INITIALIZE FROM SCRATCH
            print(f"Initializing {model_id} from SCRATCH (Random Weights)...")
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_config(
                config, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
            
            
        self.split_layer = split_layer
        self.device = self.model.device

        # Initialize and insert cross-attention
        hidden_size = self.model.config.hidden_size # 640
        num_heads = self.model.config.num_attention_heads
        self.hidden_size = hidden_size

        for i in range(split_layer, len(self.model.model.layers)):
            # Create the specific cross-attn block for this layer
            cross_attn = CrossAttention(
                dim=hidden_size,
                context_dim=hidden_size,
                num_heads=num_heads,
                dropout_rate=0.1
            )
            
            # Wrap the original layer
            original_layer = self.model.model.layers[i]
            self.model.model.layers[i] = Gemma3MultimodalLayer(
                original_layer, 
                Residual(cross_attn)
            )

        self.to(torch.bfloat16)

    def condition_image(self, image_embeds):
        """Passes image embeddings (Bs, img_q, 640) to layers 12+"""
        # Ensure we match the model's device and dtype
        self.image_embeds = image_embeds.to(next(self.parameters()).device, dtype=torch.bfloat16)
        
        for layer in self.model.model.layers:
            if isinstance(layer, Gemma3MultimodalLayer):
                layer.condition_vis_x(self.image_embeds)

    def forward(self, 
                input_ids, 
                attention_mask=None,
                return_embeddings=False,
                **kwargs):
        # HF Forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # Extraction for contrastive learning
        # Index split_layer gives the output of (split_layer - 1)
        # e.g., index 12 = output of Layer 11
        unimodal_hidden_states = outputs.hidden_states[self.split_layer]
        text_sentence_embedding = unimodal_hidden_states[:, -1, :]

        if return_embeddings:
            return outputs
        else:
            return text_sentence_embedding, outputs.logits
    
    def _lock_text(self, 
                   unlocked_layers: int = 0, 
                   freeze_layer_norm: bool = True):
        """
        Locks the unimodal encoder.
        unlocked_layers: How many unimodal layers (counting back from split_layer) to keep trainable.
        freeze_layer_norm: Whether to freeze Norm layers (RMSNorm/LayerNorm).
        """
        # 1. Ensure the Multimodal Decoder and Head are ALWAYS trainable
        for param in self.model.parameters():
            param.requires_grad = True

        # 2. Identify Unimodal components
        embeddings = self.model.model.embed_tokens
        unimodal_layer_list = self.model.model.layers[:self.split_layer]
        modules = [embeddings, *unimodal_layer_list]
        
        if unlocked_layers > 0:
            modules_to_freeze = modules[:-unlocked_layers]
        else:
            modules_to_freeze = modules

        first_unlocked_layer_idx = (len(modules) - unlocked_layers) - 1

        print(f"Locking {len(modules_to_freeze)} unimodal modules (Embeddings + Layers 0 to {first_unlocked_layer_idx - 1}).")
        print(f"Unimodal layers {max(0, first_unlocked_layer_idx)} to {self.split_layer - 1} remain trainable.")

        # 4. Perform Freezing
        for module in modules_to_freeze:
            for n, p in module.named_parameters():
                is_norm = any(x in n.split(".") for x in ["norm", "LayerNorm", "input_layernorm", "post_attention_layernorm"])
                
                if is_norm:
                    p.requires_grad = not freeze_layer_norm
                else:
                    p.requires_grad = False

    def _truncate_to_unimodal(self):
        """
        Deletes all layers from split_layer onwards, keeping only the 
        unimodal layers (0 to split_layer-1).
        """
        # 1. Physically remove the layers (indices split_layer to end)
        # This deletes the Gemma3MultimodalLayer wrappers and their weights
        self.model.model.layers = nn.ModuleList(self.model.model.layers[:self.split_layer])
        
        # 2. Update the config so the model handles the new length correctly
        # (This ensures the final layer-norm and LM-head use the correct hidden state)
        self.model.config.num_hidden_layers = self.split_layer
        
        # 3. Cleanup image references
        if hasattr(self, 'image_embeds'):
            del self.image_embeds
            
        print(f"Multimodal layers deleted. Model truncated to {self.split_layer} layers.")
