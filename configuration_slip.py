"""SLIP model configuration for HuggingFace Hub."""

from transformers import PretrainedConfig


class SLIPConfig(PretrainedConfig):
    model_type = "slip"

    def __init__(
        self,
        llm_model_name="google/gemma-3-270m",
        max_llm_len=768,
        num_img_queries=64,
        num_heads=5,
        caption_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        use_lora=False,
        unlocked_layers=4,
        split_layer=12,
        common_dim=640,
        post_train=True,
        sensor_encoder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_model_name = llm_model_name
        self.max_llm_len = max_llm_len
        self.num_img_queries = num_img_queries
        self.num_heads = num_heads
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.use_lora = use_lora
        self.unlocked_layers = unlocked_layers
        self.split_layer = split_layer
        self.common_dim = common_dim
        self.post_train = post_train
        self.sensor_encoder = sensor_encoder or {
            "embed_dim": 768,
            "num_heads": 12,
            "mlp_ratio": 4,
            "depth": 12,
            "dropout_rate": 0.1,
            "learnable_pos_emb": False,
            "max_position_embeddings": 4880,
            "patch_size": None,
            "channel_attn_type": "all_attn",
        }
