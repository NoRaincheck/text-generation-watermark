#!/usr/bin/env python3
"""CLI driver for plain greedy generation (no watermark).

Usage:

    uv run cli_greedy.py --prompt "What is gravity?" --tokens 200
"""

from colorama import Fore, init

from target_hash_gen.core import GreedyGenerator, Model, load_tokenizer
from target_hash_gen.greedy import EOS_ID, MODEL_ID

init(autoreset=True)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Plain greedy generation (no watermark)"
    )
    ap.add_argument("--prompt", type=str, default="What is gravity?")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
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

    gen = GreedyGenerator(
        model=model,
        vocab_size=tok.vocab_size,
        eos_id=EOS_ID,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    ids = gen.generate(prompt_ids, max_new_tokens=args.tokens)
    text = tok.decode(ids[len(prompt_ids) :], skip_special_tokens=True)

    print(Fore.CYAN + f"=== greedy ({args.top_k} top-k, {args.top_p} top-p) ===")
    print(text)


if __name__ == "__main__":
    main()
