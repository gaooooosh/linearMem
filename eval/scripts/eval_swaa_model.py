#!/usr/bin/env python3
"""
Custom lm-eval model wrapper for SWAA (Sliding Window Attention Adaptation) models.

This wrapper allows lm-evaluation-harness to evaluate models with SWAA configuration,
supporting various attention mechanisms and cache strategies.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path for importing swaa_patch
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from swaa_patch import SWAAConfig, hack_hf_swaa, hack_kv_cache_recurrent_state


@register_model("swaa_hf")
class SWAAHFLM(LM):
    """
    HuggingFace model with SWAA support for lm-evaluation-harness.

    This class wraps a HuggingFace CausalLM with SWAA configuration,
    allowing evaluation of sliding window attention models.
    """

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        # SWAA Configuration
        sliding_window_size: int = 2048,
        keep_first: int = 4,
        force_fa_decode: bool = False,
        non_sliding_layers: Optional[List[int]] = None,
        # Generation config
        batch_size: Optional[int] = None,
        max_length: int = 4096,
        **kwargs,
    ):
        # Handle batch_size from kwargs (lm-eval may pass it)
        batch_size = batch_size or kwargs.pop("batch_size", 1)

        # Call parent init with batch_size
        super().__init__()

        # Apply SWAA patches
        hack_kv_cache_recurrent_state()
        hack_hf_swaa(training=False)

        # Device setup
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = getattr(torch, torch_dtype)
        self.batch_size = batch_size
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            trust_remote_code=True,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            device_map={"": self.device},
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        ).eval()

        # Configure SWAA
        self.swaa_config = SWAAConfig(
            sliding_window_size=sliding_window_size,
            keep_first=keep_first,
            force_fa_decode=force_fa_decode,
            non_sliding_layers=non_sliding_layers or [],
        )
        self.model.config.swaa_config = self.swaa_config

        print(f"\n{'='*60}")
        print(f"SWAA Model Loaded: {pretrained}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.torch_dtype}")
        print(f"Attention: {attn_implementation}")
        print(f"\nSWAA Configuration:")
        print(f"  - sliding_window_size: {sliding_window_size}")
        print(f"  - keep_first: {keep_first}")
        print(f"  - force_fa_decode: {force_fa_decode}")
        print(f"  - non_sliding_layers: {non_sliding_layers}")
        print(f"{'='*60}\n")

    def tok_encode(self, string: str) -> List[int]:
        """Encode string to token ids."""
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token ids to string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _create_cache(self):
        """Create a new cache instance for each generation."""
        return DynamicCache()

    def _model_call(
        self,
        inps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            inps: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask
            past_key_values: KV cache for incremental decoding

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=inps,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        return outputs.logits

    def _model_generate(
        self,
        context: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Generate text given a context.

        Args:
            context: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Generated text
        """
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                past_key_values=self._create_cache(),
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuations given contexts.

        Args:
            requests: List of (context, continuation) tuples

        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []

        for context, continuation in requests:
            # Tokenize
            context_tokens = self.tok_encode(context)
            continuation_tokens = self.tok_encode(continuation)

            # Full sequence
            full_tokens = context_tokens + continuation_tokens
            input_ids = torch.tensor([full_tokens], device=self.device)

            # Get logits
            with torch.no_grad():
                logits = self._model_call(input_ids)

            # Compute log likelihood for continuation tokens
            cont_start = len(context_tokens)
            cont_logits = logits[0, cont_start - 1 : -1, :]  # [cont_len, vocab_size]
            cont_tokens_tensor = torch.tensor(continuation_tokens, device=self.device)

            # Calculate log probabilities
            log_probs = torch.nn.functional.log_softmax(cont_logits, dim=-1)
            token_log_probs = log_probs[range(len(continuation_tokens)), cont_tokens_tensor]

            # Sum log probabilities
            total_log_prob = token_log_probs.sum().item()

            # Check if greedy decoding would produce the continuation
            greedy_tokens = cont_logits.argmax(dim=-1)
            is_greedy = (greedy_tokens == cont_tokens_tensor).all().item()

            results.append((total_log_prob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests) -> List[float]:
        """
        Compute log-likelihood of sequences (for perplexity evaluation).

        Args:
            requests: List of sequences

        Returns:
            List of log-likelihoods
        """
        results = []

        for (sequence,) in requests:
            tokens = self.tok_encode(sequence)
            input_ids = torch.tensor([tokens], device=self.device)

            with torch.no_grad():
                logits = self._model_call(input_ids)

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
            token_log_probs = log_probs[range(len(tokens) - 1), tokens[1:]]

            results.append(token_log_probs.sum().item())

        return results

    def generate_until(self, requests) -> List[str]:
        """
        Generate text until stopping criteria.

        Args:
            requests: List of (context, until) tuples

        Returns:
            List of generated texts
        """
        results = []

        for context, until in requests:
            # Tokenize context
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

            # Prepare stopping criteria
            if isinstance(until, str):
                until = [until]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    past_key_values=self._create_cache(),
                    stop_strings=until if hasattr(self.tokenizer, "decode") else None,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove context from generated text
            if generated.startswith(context):
                generated = generated[len(context) :].strip()

            # Apply stopping criteria
            for stop in until:
                if stop in generated:
                    generated = generated.split(stop)[0].strip()

            results.append(generated)

        return results
