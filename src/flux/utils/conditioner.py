from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from typing import List
import torch


## copy from FLUX repo  

class TextConditioner(nn.Module):
    def __init__(self, version: str, max_length: int, output_type: str, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai/")
        self.version = version
        self.max_length = max_length
        self.output_key = output_type # "last_hidden_state" or "pooler_output"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: List[str], device: torch.device) -> Tensor:
        self.hf_module.to(device)
        batch_encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            ) # batch_encoding['input_ids'].shape = [bs, max_length]
        
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False, # only output last layer ouput of text model
        )
        return outputs[self.output_key] # [bs, max_length, hidden_size=4096] for xxl
        # last_hidden_state: [bs, max_length, d]
        # pooler_output: [bs, d], will do unsqueeze in pdm.py
