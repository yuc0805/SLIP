# HF architecture dict:
arch_dict = {
    # https://huggingface.co/docs/transformers/model_doc/roberta#roberta
    "roberta": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layer",
            "token_embeddings_attr": "embeddings"
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/longformer#longformer
    "longformer": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layer",
            "token_embeddings_attr": "embeddings"
        },
       # "pooler": "mean_pooler",
    },
    # gemma3_text
    "gemma3_text": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layers",
            "token_embeddings_attr": "embed_tokens"
        },

}
}