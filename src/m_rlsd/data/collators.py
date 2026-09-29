"""Collators for batching text OPD examples."""

from typing import Any

import torch
from torch.utils.data import default_collate

from m_rlsd.data.text_dataset import TextOPDExample


class TextOPDCollator:
    """Collate text OPD examples with tokenizer."""
    
    def __init__(self, tokenizer: Any, max_prompt_length: int = 512, max_response_length: int = 512):
        """Initialize collator.
        
        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
            Tokenizer (e.g., from transformers)
        max_prompt_length : int
            Max tokens in prompt
        max_response_length : int
            Max tokens in response/privileged_context
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, batch: list[TextOPDExample]) -> dict[str, Any]:
        """Collate a batch of examples.
        
        Returns
        -------
        dict with keys:
            - ids: list of example IDs
            - prompt_input_ids: [B, L_prompt]
            - prompt_attention_mask: [B, L_prompt]
            - privileged_input_ids: [B, L_context]
            - privileged_attention_mask: [B, L_context]
            - answer: list of answer strings
        """
        ids = [ex.id for ex in batch]
        answers = [ex.answer for ex in batch]
        
        # Tokenize prompts
        prompts = [ex.prompt for ex in batch]
        prompt_encoded = self.tokenizer(
            prompts,
            max_length=self.max_prompt_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize privileged context
        contexts = [ex.privileged_context for ex in batch]
        context_encoded = self.tokenizer(
            contexts,
            max_length=self.max_response_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "ids": ids,
            "prompt_input_ids": prompt_encoded["input_ids"],
            "prompt_attention_mask": prompt_encoded["attention_mask"],
            "privileged_input_ids": context_encoded["input_ids"],
            "privileged_attention_mask": context_encoded["attention_mask"],
            "answers": answers,
        }
