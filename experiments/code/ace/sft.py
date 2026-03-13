from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments


@dataclass
class SFTExample:
    prompt: str
    completion: str


class SFTDataset(Dataset):
    def __init__(self, tokenizer, examples: List[SFTExample], max_seq_len: int) -> None:
        self.tok = tokenizer
        self.examples = examples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        full = ex.prompt + ex.completion

        enc_full = self.tok(
            full,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc_full["input_ids"][0]
        attention_mask = enc_full["attention_mask"][0]

        # Mask prompt tokens in labels (train only on completion)
        enc_prompt = self.tok(
            ex.prompt,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors="pt",
        )
        prompt_len = enc_prompt["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def sft_update(
    model,
    tokenizer,
    examples: List[SFTExample],
    output_dir: str,
    max_seq_len: int,
    microbatch_size: int,
    grad_accum_steps: int,
    lr: float,
    epochs: int,
    bf16: bool,
) -> None:
    if not examples:
        return

    ds = SFTDataset(tokenizer, examples, max_seq_len=max_seq_len)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=microbatch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        bf16=bf16 and torch.cuda.is_available(),
        fp16=(not bf16) and torch.cuda.is_available(),
    )

    model.train()
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    model.eval()
