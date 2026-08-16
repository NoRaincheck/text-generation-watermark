"""Token-level watermarking on LFM2.5-350M.

Mirrors watermark_synthid.py but generates with the transformers/PyTorch model
directly. Every token gets an intrinsic color (green/red) from a keyed hash of
the token id alone; generation boosts green tokens only among near-equal
options. Because color never depends on position, the hash survives splitting.

A Watermark for Large Language Models (ICML 2023)
https://arxiv.org/pdf/2301.10226
"""

import numpy as np

from target_hash_gen.core import WatermarkGenerator, g_score, _model, _tok, EOS_ID

GREEN_FRACTION = 0.5


class BoostWatermarkGenerator(WatermarkGenerator):
    """Watermark strategy: boost green tokens in the near-equal logit band."""

    def __init__(
        self,
        seed: str | None = None,
        top_k: int = 20,
        top_p: float = 0.95,
        delta: float = 1.0,
    ) -> None:
        super().__init__(seed=seed, top_k=top_k, top_p=top_p)
        self.delta = delta

    def _sample_with_watermark(self, logits: np.ndarray, cand: np.ndarray) -> int:
        best = cand[np.argmax(logits[cand])]
        green = np.array([g_score(self.seed, int(c)) for c in cand])
        if cand.size > 1 and green.any():
            boosted = logits[cand].copy()
            boosted[green] += self.delta
            boosted[~green] -= self.delta
            probs = np.exp(boosted)
            return int(self.rng.choice(cand, p=probs / probs.sum()))
        return int(best)

    def check_hash(self, ids: list[int], seed: str | None = None) -> float:
        """Green fraction's z-score vs the expected 0.5, on any token span."""
        s = self.seed or seed
        if s is None:
            raise ValueError("seed cannot be None when check_hash")
        if not ids:
            return 0.0
        z = (sum(g_score(s, t) for t in ids) / len(ids) - GREEN_FRACTION) / np.sqrt(
            GREEN_FRACTION * (1 - GREEN_FRACTION) / len(ids)
        )
        return float(z)
