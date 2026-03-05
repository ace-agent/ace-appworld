from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

try:
    from peft import LoraConfig, get_peft_model
except Exception as e:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None

class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = stop_ids
        self.n = len(stop_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids: [batch, seq]
        if input_ids.shape[1] < self.n:
            return False
        return input_ids[0, -self.n:].tolist() == self.stop_ids

@dataclass
class HFPolicy:
    """A minimal HF policy wrapper for generation + optional LoRA training."""

    model_name: str
    trainable_lora: bool = False
    bf16: bool = True
    device: str = "cuda"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.bf16 and torch.cuda.is_available() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device.startswith("cuda") and torch.cuda.is_available() else None,
        )

        if self.trainable_lora:
            if LoraConfig is None or get_peft_model is None:
                raise ImportError("peft is required for trainable_lora=True. Install peft.")
            lora_cfg = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(self.lora_target_modules),
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.train()
        else:
            self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        #messages = [
        #    {"role": "user", "content": prompt}
        #]

        #inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        inputs = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,  return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        #stop_str = "</json>"
        #stop_ids = self.tokenizer.encode(stop_str, add_special_tokens=False)
        #stopping = StoppingCriteriaList([StopOnSubsequence(stop_ids)])

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature is not None and temperature > 0),
            temperature=0.0, 
            top_p=float(top_p),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            #stopping_criteria=stopping,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        #if text.startswith(prompt):
        #    return text[len(prompt):].strip()
        return text.strip()
