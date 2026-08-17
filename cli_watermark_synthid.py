#!/usr/bin/env python3
"""CLI driver for SynthID-Text-style tournament watermarking.

Usage:

    uv run cli_watermark_synthid.py --prompt "What is gravity?" --tokens 200
"""

import numpy as np

from colorama import Fore, Style

from target_hash_gen.core import Model, g_score, load_tokenizer, DEFAULT_SEED, _tok
from target_hash_gen.greedy import GreedyGenerator
from target_hash_gen.watermark_synthid import TournamentWatermarkGenerator


def hit_strength(seed: str, token_id: int, m: int) -> int:
    """How many of the m watermarking functions score this token green (0..m)."""
    return sum(g_score(seed, token_id, str(layer)) for layer in range(m))


def ref_quantiles(ids: list[int], seed: str, m: int) -> tuple[float, float, float]:
    """p50/p75/p90 of hit_strength across a reference text (e.g. the `wm`
    output), used to derive dynamic coloring thresholds."""
    hits = [hit_strength(seed, int(t), m) for t in ids]
    return tuple(np.percentile(hits, [50, 75, 90]))  # type: ignore[return-value]


def hit_color(hits: int, p50: float, p75: float, p90: float) -> str:
    """Color a token by its hit strength relative to the reference's quantile
    bins: no color below p50, green in [p50, p75), yellow in [p75, p90),
    red at or above p90."""
    if hits < p50:
        return str(Style.RESET_ALL)
    if hits < p75:
        return str(Fore.GREEN)
    if hits < p90:
        return str(Fore.YELLOW)
    return str(Fore.RED)


def colored_text(
    ids: list[int], tok, seed: str, m: int = 5, p50: float = 0.0, p75: float = 0.0, p90: float = 0.0
) -> str:
    """Decode tokens colored by hit_strength into the reference text's quantile
    bins. Defaults mimic m watermarking layers: no color below p50 (2/3 of m),
    green [p50, p75), yellow [p75, p90), red at/above p90."""
    if p50 == 0 and p75 == 0 and p90 == 0:
        p50, p75, p90 = m * 2 / 3, m * 3 / 4, m * 0.9
    parts: list[str] = []
    for t in ids:
        text = tok.decode([t], skip_special_tokens=True)
        hits = hit_strength(seed, int(t), m)
        parts.append(hit_color(hits, p50, p75, p90) + text)
    return "".join(parts)


def split_starts(tokens: list[int], tok) -> list[int]:
    """Token indices (1-based) of sentence starts, skipping too-short tails."""
    return [i + 1 for i, t in enumerate(tokens) if tok.decode([t]) in (".", "!", "?") and len(tokens[i + 1 :]) >= 16]


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Invisible content hash via SynthID-Text-style tournament")
    ap.add_argument("--prompt", type=str, default="What is gravity?")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument(
        "--layers",
        type=int,
        default=5,
        metavar="M",
        help="tournament layers / watermarking functions (default 5)",
    )
    ap.add_argument(
        "--competitors",
        type=int,
        default=2,
        metavar="K",
        help="tokens per match (reserved; matches are currently pairwise) (default 2)",
    )
    ap.add_argument("--seed", default="a seed")
    ap.add_argument("--wrong-seed", default="negative key")
    args = ap.parse_args(argv)

    messages = [
        {
            "role": "user",
            "content": args.prompt,
        },
    ]
    prompt = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = _tok(prompt, add_special_tokens=False)["input_ids"]

    gen_wm = TournamentWatermarkGenerator(
        seed=args.seed,
        m=args.layers,
        k=args.competitors,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    gen_neg = TournamentWatermarkGenerator(
        seed=args.wrong_seed,
        m=args.layers,
        k=args.competitors,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    gen_plain = GreedyGenerator(
        top_k=args.top_k,
        top_p=args.top_p,
    )

    wm = gen_wm.generate(prompt_ids, max_new_tokens=args.tokens)
    neg = gen_neg.generate(prompt_ids, max_new_tokens=args.tokens)
    plain = gen_plain.generate(prompt_ids, max_new_tokens=args.tokens)
    wm, neg, plain = (ids[len(prompt_ids) :] for ids in (wm, neg, plain))

    print(
        f"{Fore.CYAN}=== watermarked output ({args.layers} layers, {args.competitors} competitors/match) ==={Style.RESET_ALL}"
    )
    p50, p75, p90 = ref_quantiles(wm, args.seed, args.layers)
    title = f"(colored vs watermarked ref: p50={p50:.2f}, p75={p75:.2f}, p90={p90:.2f})"
    print(f"{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(colored_text(wm, _tok, args.seed, args.layers, p50, p75, p90))
    print(f"{Fore.CYAN}\n=== negative-seed output ==={Style.RESET_ALL}")
    print(colored_text(neg, _tok, args.seed, args.layers, p50, p75, p90))
    print(f"{Fore.CYAN}\n=== plain output (baseline) ==={Style.RESET_ALL}")
    print(colored_text(plain, _tok, args.seed, args.layers, p50, p75, p90))

    print(f"{Fore.CYAN}\n=== detection ==={Style.RESET_ALL}")
    print(f"watermarked, correct key  : {gen_wm.check_hash(wm, args.seed)}")
    print(f"watermarked, wrong key    : {gen_wm.check_hash(wm, args.wrong_seed)}")
    print(f"negative-seed, correct key: {gen_wm.check_hash(neg, args.seed)}")
    print(f"plain, correct key        : {gen_wm.check_hash(plain, args.seed)}")

    for name, text in [
        ("watermarked", wm),
        ("negative-seed", neg),
        ("plain", plain),
    ]:
        print(f"{Fore.CYAN}\n=== hash across splits of {name} text ==={Style.RESET_ALL}")
        print(f"{'from tok':>8} | {'seed':<32}")
        for st in split_starts(text, _tok):
            span = text[st:]
            print(f"{st:>8} | {gen_wm.check_hash(span, args.seed):<32} ")


if __name__ == "__main__":
    main()
