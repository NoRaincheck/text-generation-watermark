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
