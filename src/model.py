"""Thin wrapper around a small Hugging Face causal LM for batched generation.

The default model id is "sshleifer/tiny-gpt2", a very small GPT-2 variant that
loads in a second or two and is handy for wiring/tests. For slightly more
realistic output you can swap in "distilgpt2" by passing model_id or setting the
MODEL_ID environment variable.

Loading is lazy: weights are only pulled the first time generate() runs, so
importing this module stays cheap (useful for tests that mock the model).
"""

from __future__ import annotations

import os
from typing import List, Optional


class CausalLMWrapper:
    """Loads a causal LM + tokenizer once and runs padded batched generation."""

    def __init__(self, model_id: Optional[str] = None) -> None:
        # Precedence: explicit arg > env var > tiny default.
        self.model_id = model_id or os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2")
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # Imported lazily so `import src.model` does not require torch/transformers
        # to be present at import time (tests can mock generate instead).
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # GPT-2 style tokenizers have no pad token; reuse EOS so batched padding
        # works. Left padding keeps the newest tokens aligned for generation.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.to(self._device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self._device  # type: ignore[return-value]

    def generate(self, prompts: List[str], max_new_tokens: int = 32) -> List[str]:
        """Generate a continuation for each prompt in a single batched call.

        Returns only the newly generated text (the prompt is stripped off).
        """
        if not prompts:
            return []

        self._ensure_loaded()
        import torch

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # With left padding, the prompt occupies the first input_ids.shape[1]
        # columns for every row, so the continuation is everything after it.
        prompt_len = enc["input_ids"].shape[1]
        gen_tokens = out[:, prompt_len:]
        return tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
