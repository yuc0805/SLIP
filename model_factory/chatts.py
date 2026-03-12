import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoProcessor
from einops import rearrange

class ChatTSEncoder(torch.nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        **cfg,
    ):
        super().__init__()

        hf_model = 'bytedance-research/ChatTS-8B'
        model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True, device_map="cpu", torch_dtype='float32')
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(hf_model, trust_remote_code=True, tokenizer=tokenizer)

        self.model = model
        self.device = device
        self.embed_dim = 4096

    def forward(
        self,
        x,
        *args,
        **kwargs, # keep training script happy
    ):
        """
        x: numpy or torch array, shape (BS, nvar, L)
        returns: Tensor (BS, embed_dim)   embed_dim is 512 for chronos t5 small
        """
        # x still in cpu, find device and move it
        

        input_dict = {
            'input_ids': x['input_ids'].to(self.device,non_blocking=True),
            'attention_mask': x['attention_mask'].to(self.device,non_blocking=True),
            'timeseries': x['timeseries'].to(self.device,non_blocking=True)
        }
        out = self.model(**input_dict, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1]  # (BS, L, D)
        
        return hidden, None

