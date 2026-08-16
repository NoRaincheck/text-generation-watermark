"""Invisible content hash via token-level watermarking on LFM2.5-350M (PyTorch).

Mirrors watermark_synthid.py but generates with the transformers/PyTorch model
directly. Every token gets an intrinsic color (green/red) from a keyed hash of
the token id alone; generation boosts green tokens only among near-equal
options. Because color never depends on position, the hash survives splitting.

A Watermark for Large Language Models (ICML 2023)
https://arxiv.org/pdf/2301.10226
"""

import numpy as np
from colorama import Fore, init

from core import (
    WatermarkGenerator,
    _rng_for,
    g_score,
    load_tokenizer,
    Model,
)

init(autoreset=True)

MODEL_ID = "LiquidAI/LFM2.5-350M"
DEFAULT_SEED = "target-hash-gen demo key"
EOS_ID = 7
GREEN_FRACTION = 0.5


class BoostWatermarkGenerator(WatermarkGenerator):
    """Watermark strategy: boost green tokens in the near-equal logit band."""

    def __init__(
        self,
        model: Model,
        vocab_size: int,
        eos_id: int,
        seed: str | None = None,
        top_k: int = 20,
        top_p: float = 0.95,
        eps: float = 1.0,
        delta: float = 2.0,
    ) -> None:
        super().__init__(model, vocab_size, eos_id, seed, top_k, top_p)
        self.eps = eps
        self.delta = delta

    def _sample_with_watermark(self, logits: np.ndarray, cand: np.ndarray) -> int:
        best = cand[np.argmax(logits[cand])]
        near_eq = cand[logits[cand] >= logits[best] - self.eps]
        green = np.array([g_score(self.seed, int(c)) for c in near_eq])
        if near_eq.size > 1 and green.any():
            boosted = logits[near_eq].copy()
            boosted[green] += self.delta
            probs = np.exp(boosted - boosted.max())
            return int(self.rng.choice(near_eq, p=probs / probs.sum()))
        return int(best)


def check_hash(ids: list[int], seed: str = DEFAULT_SEED) -> str:
    """Green fraction and z-score vs the expected 0.5, on any token span."""
    if not ids:
        return "Hash(0/0 frac=0.000 z=+0.0)"
    green = sum(g_score(seed, t) for t in ids)
    frac = green / len(ids)
    z = (frac - GREEN_FRACTION) / np.sqrt(
        GREEN_FRACTION * (1 - GREEN_FRACTION) / len(ids)
    )
    return f"Hash({green}/{len(ids)} frac={frac:.3f} z={z:+.1f})"


def colored_hash(ids: list[int], seed: str) -> str:
    """check_hash() colorized: green when the watermark is detected (z >= 3),
    red when it isn't."""
    n = len(ids) or 1
    green = sum(g_score(seed, t) for t in ids) if ids else 0
    z = (green / n - GREEN_FRACTION) / np.sqrt(
        GREEN_FRACTION * (1 - GREEN_FRACTION) / n
    )
    return (Fore.GREEN if z >= 1.65 else Fore.RED) + check_hash(ids, seed)


def split_starts(tokens: list[int], tok) -> list[int]:
    """Token indices (1-based) of sentence starts, skipping too-short tails."""
    return [
        i + 1
        for i, t in enumerate(tokens)
        if tok.decode([t]) in (".", "!", "?") and len(tokens[i + 1 :]) >= 16
    ]


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Invisible content hash via token-level watermarking"
    )
    ap.add_argument("--prompt", type=str, default="What is gravity?")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=20)
    for name, default in (("top-p", 0.95), ("eps", 1.0), ("delta", 2.0)):
        ap.add_argument(f"--{name}", type=float, default=default)
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--wrong-seed", default="negative key")
    args = ap.parse_args(argv)

    tok = load_tokenizer(MODEL_ID)
    model = Model(MODEL_ID)
    messages = [
        {
            "role": "user",
            "content": args.prompt,
        },
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]

    gen_wm = BoostWatermarkGenerator(
        model=model,
        vocab_size=tok.vocab_size,
        eos_id=EOS_ID,
        seed=args.seed,
        top_k=args.top_k,
        top_p=args.top_p,
        eps=args.eps,
        delta=args.delta,
    )
    gen_neg = BoostWatermarkGenerator(
        model=model,
        vocab_size=tok.vocab_size,
        eos_id=EOS_ID,
        seed=args.wrong_seed,
        top_k=args.top_k,
        top_p=args.top_p,
        eps=args.eps,
        delta=args.delta,
    )
    gen_plain = BoostWatermarkGenerator(
        model=model,
        vocab_size=tok.vocab_size,
        eos_id=EOS_ID,
        seed=None,  # no watermark
        top_k=args.top_k,
        top_p=args.top_p,
        eps=args.eps,
        delta=args.delta,
    )

    wm = gen_wm.generate(prompt_ids, max_new_tokens=args.tokens)
    neg = gen_neg.generate(prompt_ids, max_new_tokens=args.tokens)
    plain = gen_plain.generate(prompt_ids, max_new_tokens=args.tokens)
    wm, neg, plain = (ids[len(prompt_ids) :] for ids in (wm, neg, plain))

    print(Fore.CYAN + "=== watermarked output ===")
    print(tok.decode(wm, skip_special_tokens=True))
    print(Fore.CYAN + "\n=== negative-seed output ===")
    print(tok.decode(neg, skip_special_tokens=True))
    print(Fore.CYAN + "\n=== plain output (baseline) ===")
    print(tok.decode(plain, skip_special_tokens=True))

    print(Fore.CYAN + "\n=== detection ===")
    print(f"{Fore.GREEN}watermarked, correct key  : {colored_hash(wm, args.seed)}")
    print(f"{Fore.RED}watermarked, wrong key    : {colored_hash(wm, args.wrong_seed)}")
    print(f"{Fore.RED}negative-seed, correct key: {colored_hash(neg, args.seed)}")
    print(f"{Fore.RED}plain, correct key        : {colored_hash(plain, args.seed)}")

    for name, text, key, wrong in [
        ("watermarked", wm, args.seed, args.wrong_seed),
        ("negative-seed", neg, args.wrong_seed, args.seed),
        ("plain", plain, args.seed, args.wrong_seed),
    ]:
        print(f"{Fore.CYAN}\n=== hash across splits of {name} text ===")
        print(f"{'from tok':>8} | {'key':<32} {'neg key':<32}")
        for st in split_starts(text, tok):
            span = text[st:]
            print(
                f"{st:>8} | "
                f"{colored_hash(span, key):<32} "
                f"{colored_hash(span, wrong):<32}"
            )


if __name__ == "__main__":
    main()
