#!/usr/bin/env python3
"""CLI driver for token-level watermarking (boost strategy).

Usage:

    uv run cli_watermark.py --prompt "What is gravity?" --tokens 200
"""

from colorama import Fore, Style

from target_hash_gen.core import Model, g_score, load_tokenizer
from target_hash_gen.greedy import GreedyGenerator
from target_hash_gen.watermark import (
    BoostWatermarkGenerator,
    EOS_ID,
    MODEL_ID,
    DEFAULT_SEED,
)


def colored_text(ids: list[int], tok, seed: str) -> str:
    """Decode tokens with each colored green (green-list) or red (not)."""
    parts: list[str] = []
    for t in ids:
        is_green = g_score(seed, int(t))
        text = tok.decode([t], skip_special_tokens=True)
        parts.append((Fore.GREEN if is_green else Style.RESET_ALL) + text)
    return "".join(parts)


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
    for name, default in (("top-p", 0.95), ("delta", 1.0)):
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
        delta=args.delta,
    )
    gen_neg = BoostWatermarkGenerator(
        model=model,
        vocab_size=tok.vocab_size,
        eos_id=EOS_ID,
        seed=args.wrong_seed,
        top_k=args.top_k,
        top_p=args.top_p,
        delta=args.delta,
    )
    gen_plain = GreedyGenerator(
        model=model,
        vocab_size=tok.vocab_size,
        eos_id=EOS_ID,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    wm = gen_wm.generate(prompt_ids, max_new_tokens=args.tokens)
    neg = gen_neg.generate(prompt_ids, max_new_tokens=args.tokens)
    plain = gen_plain.generate(prompt_ids, max_new_tokens=args.tokens)
    wm, neg, plain = (ids[len(prompt_ids) :] for ids in (wm, neg, plain))

    print(Fore.CYAN + "=== watermarked output ===" + Style.RESET_ALL)
    print(colored_text(wm, tok, args.seed))
    print(Fore.CYAN + "\n=== negative-seed output ===" + Style.RESET_ALL)
    print(colored_text(neg, tok, args.seed))
    print(Fore.CYAN + "\n=== plain output (baseline) ===" + Style.RESET_ALL)
    print(colored_text(plain, tok, args.seed))

    print(Fore.CYAN + "\n=== detection ===" + Style.RESET_ALL)
    print(f"watermarked, correct key  : {gen_wm.check_hash(wm, args.seed)}")
    print(f"watermarked, wrong key    : {gen_wm.check_hash(wm, args.wrong_seed)}")
    print(f"negative-seed, correct key: {gen_wm.check_hash(neg, args.seed)}")
    print(f"plain, correct key        : {gen_wm.check_hash(plain, args.seed)}")

    for name, text in [
        ("watermarked", wm),
        ("negative-seed", neg),
        ("plain", plain),
    ]:
        print(
            f"{Fore.CYAN}\n=== hash across splits of {name} text ===" + Style.RESET_ALL
        )
        print(f"{'from tok':>8} | {'seed':<32}")
        for st in split_starts(text, tok):
            span = text[st:]
            print(f"{st:>8} | {gen_wm.check_hash(span, args.seed):<32} ")


if __name__ == "__main__":
    main()
