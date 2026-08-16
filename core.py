"""Core infrastructure: model loading, tokenizers, RNG, nucleus & dist helpers.

Shared between watermark.py and watermark_synthid.py so the scripts stay
minimal and readable for teaching purposes.
"""

import hashlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _rng_for(seed: str) -> np.random.Generator:
    """Deterministic generator derived from the watermark key string."""
    h = hashlib.blake2b(digest_size=8, key=seed.encode()[:64])
    return np.random.default_rng(int.from_bytes(h.digest(), "big"))


class Model:
    """PyTorch model wrapper with manual KV-cache decoding."""

    def __init__(self, model_id: str, device=None, dtype=torch.bfloat16) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device is None else None,
            dtype=dtype,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def run(self, ids: np.ndarray, past_key_values=None) -> np.ndarray:
        x = torch.tensor(ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(x, use_cache=True, past_key_values=past_key_values)
        return out.logits[0, -1].float().cpu().numpy(), out.past_key_values


def load_tokenizer(model_id: str, local_files_only: bool = True):
    """Load the tokenizer for the given model."""
    return AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)


def _nucleus(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    """Top-k then nucleus filter; returns the surviving token indices."""
    indices = (
        np.argpartition(logits, -top_k)[-top_k:]
        if top_k > 0
        else np.arange(len(logits))
    )
    order = np.argsort(logits[indices])[::-1]
    soft = np.exp(logits[indices[order]] - logits[indices[order]].max())
    soft /= soft.sum()
    mask = np.cumsum(soft) <= top_p
    if not mask.any():
        mask[0] = True
    return indices[order][mask]


def g_score(seed: str, token_id: int, salt: str | None = None) -> bool:
    """Score (0/1) assigned to a token by a keyed blake2b hash.

    If *salt* is provided the key is ``f"{seed}_{salt}"`` so each
    watermarking layer gets an independent hash function.
    """
    key = f"{seed}_{salt}".encode()[:64] if salt is not None else seed.encode()[:64]
    h = hashlib.blake2b(digest_size=8, key=key)
    h.update(int(token_id).to_bytes(4, "big"))
    return bool(h.digest()[0] & 1)


def _dist(logits: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Softmax probabilities over the candidate tokens in `cand`."""
    soft = np.exp(logits[cand] - logits[cand].max())
    return soft / soft.sum()


class WatermarkGenerator:
    """Base class for watermark-aware text generation.

    Owns the shared boilerplate — KV-cache management, nucleus filtering,
    EOS handling, and deterministic RNG — so subclasses only need to
    implement ``_sample_with_watermark`` to define their watermark strategy.

    For plain (non-watermarked) greedy generation, use :class:`GreedyGenerator`
    in *greedy.py* instead.
    """

    def __init__(
        self,
        model: Model,
        vocab_size: int,
        eos_id: int,
        seed: str | None = None,
        top_k: int = 20,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.seed = seed
        self.top_k = top_k
        self.top_p = top_p
        self.rng = _rng_for(seed) if seed is not None else None

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        """Sample a continuation, applying the watermark strategy when seeded."""
        ids = list(prompt_ids)
        arr = np.array([ids], dtype=np.int64)
        past_key_values = None

        for _ in range(max_new_tokens):
            logits, past_key_values = self.model.run(arr, past_key_values)
            cand = _nucleus(logits[: self.vocab_size], self.top_k, self.top_p)

            if self.seed is not None:
                pick = self._sample_with_watermark(logits, cand)
            else:
                pick = int(cand[np.argmax(logits[cand])])

            ids.append(pick)
            if pick == self.eos_id:
                break
            arr = np.array([[pick]], dtype=np.int64)

        return ids

    def _sample_with_watermark(self, logits: np.ndarray, cand: np.ndarray) -> int:
        """Override in subclass to define the watermark sampling strategy."""
        raise NotImplementedError


class GreedyGenerator:
    """Plain greedy generation with nucleus sampling — no watermark.

    Uses the same KV-cache loop and nucleus filtering as
    :class:`WatermarkGenerator`, but always picks the highest-logit
    candidate. Useful as a baseline or when you want generation without
    any watermarking overhead.
    """

    def __init__(
        self,
        model: Model,
        vocab_size: int,
        eos_id: int,
        top_k: int = 20,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.top_k = top_k
        self.top_p = top_p

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        """Greedy sample: pick the highest-logit token from the nucleus."""
        ids = list(prompt_ids)
        arr = np.array([ids], dtype=np.int64)
        past_key_values = None

        for _ in range(max_new_tokens):
            logits, past_key_values = self.model.run(arr, past_key_values)
            cand = _nucleus(logits[: self.vocab_size], self.top_k, self.top_p)
            pick = int(cand[np.argmax(logits[cand])])

            ids.append(pick)
            if pick == self.eos_id:
                break
            arr = np.array([[pick]], dtype=np.int64)

        return ids
