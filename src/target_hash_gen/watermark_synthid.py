"""Invisible content hash via SynthID-Text-style tournament watermarking.

A simplified version of the tournament-sampling scheme from

    Scalable watermarking for identifying large language model outputs
    https://www.nature.com/articles/s41586-024-08025-4

The paper's generative scheme has three swappable components (seed generator,
sampling algorithm, scoring function). We keep the tournament sampling algorithm
and the mean-g scoring function, but simplify the seed generator to depend only
on the token id (not a sliding context window) so the hash stays split-invariant:
any sub-span scores exactly like the full text.

Watermarking functions: m independent keyed hashes g_l(seed, layer, token_id)
assign a 0/1 score to each candidate token. To keep the seeds simple, layer n
uses the key "{BASE_SEED}_layer{n}" (a cheap way to get independent functions).

Tournament sampling is played as a literal elimination bracket over the
nucleus-filtered candidates: each layer, the surviving tokens are shuffled and
paired up, and each pair advances the token with the higher g-score (ties broken
by a coin flip). After m layers the final token is the sole survivor, or is
chosen uniformly at random among the survivors when more than one remains.

Detection: the mean g-value over m*len(text) Bernoulli observations. Plain text
scores ~0.5; tournament output is biased toward higher g-values, so the z-score
against the 0.5 null grows with sqrt(m*len).
"""

import numpy as np

from target_hash_gen.core import WatermarkGenerator, g_score, _model, _tok, EOS_ID

GREEN_FRACTION = 0.5


def tournament_sample(
    logits: np.ndarray,
    cand: np.ndarray,
    m: int,
    k: int,
    seed: str,
    rng: np.random.Generator,
) -> int:
    """Tournament sampling as an elimination bracket.

    Each layer, the surviving candidates are shuffled and paired up; within a
    pair the token with the higher g-score advances, and ties are broken by a
    coin flip (an odd leftover advances unopposed). After ``m`` layers the token
    is the sole survivor, or a uniform random draw among the survivors when more
    than one remains. ``k`` is reserved for the paper's k-way variant and is
    currently ignored (matches are always pairwise)."""
    survivors = list(cand)
    for layer in range(m):
        if len(survivors) <= 1:
            break
        rng.shuffle(survivors)
        next_round = []
        for i in range(0, len(survivors) - 1, 2):
            a, b = survivors[i], survivors[i + 1]
            ga = g_score(seed, int(a), layer)
            gb = g_score(seed, int(b), layer)
            if ga != gb:
                next_round.append(a if ga else b)
            elif rng.random() < 0.5:
                next_round.append(a)
            else:
                next_round.append(b)
        if len(survivors) % 2:
            next_round.append(survivors[-1])
        survivors = next_round
    if len(survivors) == 1:
        return int(survivors[0])
    return int(rng.choice(survivors))


class TournamentWatermarkGenerator(WatermarkGenerator):
    """Watermark strategy: tournament sampling with m layers, k competitors."""

    def __init__(
        self,
        seed: str | None = None,
        m: int = 5,
        k: int = 2,
        top_k: int = 20,
        top_p: float = 0.95,
    ) -> None:
        super().__init__(seed=seed, top_k=top_k, top_p=top_p)
        self.m = m
        self.k = k

    def _sample_with_watermark(self, logits: np.ndarray, cand: np.ndarray) -> int:
        return tournament_sample(logits, cand, self.m, self.k, self.seed, self.rng)

    def check_hash(self, ids: list[int], seed: str | None = None) -> float:
        """Z-score vs the expected 0.5 (mean g-score over m layers per token,
        seed per layer = "{seed}_layer{n}") on any token span."""
        s = seed or self.seed
        if s is None:
            raise ValueError("seed cannot be None when check_hash")
        n = len(ids) * self.m
        if n == 0:
            return 0.0
        frac = sum(g_score(s, t, layer) for t in ids for layer in range(self.m)) / n
        z = (frac - GREEN_FRACTION) / np.sqrt(GREEN_FRACTION * (1 - GREEN_FRACTION) / n)
        return float(z)
