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
from datetime import datetime
from typing import Optional, List, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from swaa_patch import SWAAConfig, hack_hf_swaa, hack_kv_cache_recurrent_state

# Global log file path
EVAL_LOG_FILE = Path(__file__).parent.parent / "evaluation.log"


def _init_log_file():
    """Initialize a fresh log file at the start of evaluation."""
    with open(EVAL_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Evaluation Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")


def _log_sample(method: str, idx: int, input_len: int, output: str, extra_info: str = ""):
    """Log a single sample's information."""
    with open(EVAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{method}] Sample #{idx}\n")
        f.write(f"  Input Length: {input_len} tokens\n")
        if extra_info:
            f.write(f"  {extra_info}\n")
        f.write(f"  Output: {output[:500]}{'...' if len(output) > 500 else ''}\n")
        f.write(f"  {'-'*60}\n")


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
        # Memory optimization for long sequences
        max_chunk_size: int = 2048,
        **kwargs,
    ):
        # Handle batch_size from kwargs (lm-eval may pass it)
        batch_size = batch_size or kwargs.pop("batch_size", 1)

        # Parse non_sliding_layers from string if needed (lm-eval passes it as string)
        if isinstance(non_sliding_layers, str):
            import ast
            try:
                non_sliding_layers = ast.literal_eval(non_sliding_layers)
            except (ValueError, SyntaxError):
                non_sliding_layers = []

        # Ensure non_sliding_layers is a list
        if non_sliding_layers is None:
            non_sliding_layers = []

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
        self.max_chunk_size = max_chunk_size

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
            non_sliding_layers=non_sliding_layers,
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

        # Initialize fresh evaluation log
        _init_log_file()

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
            requests: List of Instance objects with arguments = (context, continuation)

        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []

        pbar = tqdm(enumerate(requests), total=len(requests), desc="loglikelihood", leave=True)
        for idx, request in pbar:
            # Extract arguments from Instance object
            context, continuation = request.arguments

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

            # Log sample
            _log_sample(
                method="loglikelihood",
                idx=idx,
                input_len=len(full_tokens),
                output=f"log_prob={total_log_prob:.4f}, is_greedy={is_greedy}",
                extra_info=f"Context: {len(context_tokens)} tokens | Continuation: {len(continuation_tokens)} tokens"
            )

        return results

    def loglikelihood_rolling(self, requests) -> List[float]:
        """
        Compute log-likelihood of sequences (for perplexity evaluation).

        Args:
            requests: List of Instance objects with arguments = (sequence,)

        Returns:
            List of log-likelihoods
        """
        results = []

        pbar = tqdm(enumerate(requests), total=len(requests), desc="loglikelihood_rolling", leave=True)
        for idx, request in pbar:
            # Extract arguments from Instance object
            (sequence,) = request.arguments

            tokens = self.tok_encode(sequence)

            # Chunked processing to avoid OOM on long sequences
            total_log_prob = 0.0
            chunk_size = self.max_chunk_size

            for chunk_start in range(0, len(tokens), chunk_size):
                chunk_end = min(chunk_start + chunk_size + 1, len(tokens))  # +1 for next token prediction
                chunk_tokens = tokens[chunk_start:chunk_end]

                if len(chunk_tokens) < 2:
                    continue

                input_ids = torch.tensor([chunk_tokens], device=self.device)

                with torch.no_grad():
                    logits = self._model_call(input_ids)

                # Compute log probabilities for this chunk
                log_probs = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
                chunk_token_log_probs = log_probs[range(len(chunk_tokens) - 1), chunk_tokens[1:]]

                total_log_prob += chunk_token_log_probs.sum().item()

                # Clear cache to free memory
                torch.cuda.empty_cache()

            results.append(total_log_prob)

            # Log sample
            _log_sample(
                method="loglikelihood_rolling",
                idx=idx,
                input_len=len(tokens),
                output=f"log_prob={total_log_prob:.4f}",
                extra_info=f"Chunks: {(len(tokens) + chunk_size - 1) // chunk_size}"
            )

        return results

    def generate_until(self, requests) -> List[str]:
        """
        Generate text until stopping criteria.

        Args:
            requests: List of Instance objects with arguments = (context, until)

        Returns:
            List of generated texts
        """
        results = []

        pbar = tqdm(enumerate(requests), total=len(requests), desc="generate_until", leave=True)
        for idx, request in pbar:
            # Extract arguments from Instance object
            context, until = request.arguments

            # Tokenize context
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]

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
                    stop_strings=until if until else None,
                    tokenizer=self.tokenizer,
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

            # Log sample
            output_len = outputs.shape[1] - input_len
            _log_sample(
                method="generate_until",
                idx=idx,
                input_len=input_len,
                output=generated,
                extra_info=f"Output Length: {output_len} tokens | Stop: {until}"
            )

        return results
