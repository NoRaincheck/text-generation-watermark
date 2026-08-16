"""Plain greedy generation with nucleus sampling — no watermark.

Demonstrates how :class:`GreedyGenerator` from core.py is used for
non-watermarked baseline generation. This file is kept separate to show
the pattern: a script owns its CLI, model config, and output formatting,
while the generator class owns the sampling loop.

Usage:

    uv run greedy.py --prompt "What is gravity?" --tokens 200
"""

from colorama import Fore, init

from core import GreedyGenerator, Model, load_tokenizer

init(autoreset=True)

MODEL_ID = "LiquidAI/LFM2.5-350M"
EOS_ID = 7


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
