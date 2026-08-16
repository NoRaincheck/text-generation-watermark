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

Tournament sampling is implemented in the paper's closed form (its vectorized
update_scores): instead of literally drawing k**m tokens and playing matches,
each layer reweights the next-token distribution, boosting tokens whose g_l = 1
by multiplying their probability by (1 + g_l - g_mass_l) when k = 2 (the
non-distortionary configuration), or by the closed-form coefficients for k > 2
(the distortionary variant). The final token is resampled from the reweighted
distribution. All resampling/boosting happens inside the tournament; the
generation loop only feeds it the nucleus-filtered candidates.

Detection: the mean g-value over m*len(text) Bernoulli observations. Plain text
scores ~0.5; tournament output is biased toward higher g-values, so the z-score
against the 0.5 null grows with sqrt(m*len).
"""

import hashlib

import numpy as np
import torch
from colorama import Fore, init
from transformers import AutoModelForCausalLM, AutoTokenizer

init(autoreset=True)

MODEL_ID = "LiquidAI/LFM2.5-350M"
DEFAULT_SEED = "target-hash-gen demo key"
EOS_ID = 7


def g_score(seed: str, layer: int, token_id: int) -> bool:
    """Score (0/1) assigned to a token by watermarking function layer `layer`.

    Layer n derives its key from "{seed}_layer{n}", so the m functions are
    independent pseudorandom hashes. Depends only on (key, layer, token_id),
    never on position or context, so the watermark survives splitting."""
    key = f"{seed}_layer{layer}".encode()[:64]
    h = hashlib.blake2b(digest_size=8, key=key)
    h.update(int(token_id).to_bytes(4, "big"))
    return bool(h.digest()[0] & 1)


def _rng_for(seed: str) -> np.random.Generator:
    """Deterministic generator derived from the watermark key string."""
    h = hashlib.blake2b(digest_size=8, key=seed.encode()[:64])
    return np.random.default_rng(int.from_bytes(h.digest(), "big"))


class Model:
    """PyTorch folks model wrapper with manual KV-cache decoding."""

    def __init__(self, device=None, dtype=torch.bfloat16) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto" if device is None else None,
            dtype=dtype,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def run(self, ids: np.ndarray, past_key_values=None) -> np.ndarray:
        import torch

        x = torch.tensor(ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(x, use_cache=True, past_key_values=past_key_values)
        return out.logits[0, -1].float().cpu().numpy(), out.past_key_values


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


def _dist(logits: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Softmax probabilities over the candidate tokens in `cand`."""
    soft = np.exp(logits[cand] - logits[cand].max())
    return soft / soft.sum()


def tournament_sample(
    logits: np.ndarray,
    cand: np.ndarray,
    m: int,
    k: int,
    seed: str,
    rng: np.random.Generator,
) -> int:
    """Tournament sampling in the paper's closed form (its update_scores).

    Each layer reweights the distribution over the nucleus candidates, boosting
    tokens whose g_l = 1: with k = 2 (non-distortionary) each candidate's
    probability is multiplied by (1 + g_l - g_mass_l); for k > 2 the paper's
    distortionary coefficients are used. The final token is resampled from the
    reweighted distribution. All boosting/resampling happens here, inside the
    tournament."""
    probs = _dist(logits, cand)
    for layer in range(m):
        g = np.array([g_score(seed, layer, int(t)) for t in cand], dtype=float)
        g_mass = float((g * probs).sum())
        if k == 2:
            probs = probs * (1.0 + g - g_mass)
        else:
            coeff_not_in_g = (1.0 - g_mass) ** (k - 1)
            coeff_in_g = (
                (1.0 - (1.0 - g_mass) ** k) / g_mass if g_mass > 0 else 1.0
            )
            probs = probs * np.where(g > 0, coeff_in_g, coeff_not_in_g)
        probs /= probs.sum()
    return int(rng.choice(cand, p=probs))


def generate(
    prompt_ids: list[int],
    max_new_tokens: int,
    model: Model,
    vocab_size: int,
    seed: str | None = None,
    m: int = 5,
    k: int = 2,
    top_k: int = 20,
    top_p: float = 0.95,
) -> list[int]:
    """Sample continuations; with a seed, run the tournament so the winner tends
    to score high under the m watermarking functions. Seeded runs use a generator
    derived from the key so output is reproducible; plain runs (seed=None) are
    greedy over the nucleus."""
    ids = list(prompt_ids)
    arr = np.array([ids], dtype=np.int64)
    past_key_values = None
    rng = _rng_for(seed) if seed is not None else None
    for _ in range(max_new_tokens):
        logits, past_key_values = model.run(arr, past_key_values)
        cand = _nucleus(logits[:vocab_size], top_k, top_p)
        if seed is not None:
            pick = tournament_sample(logits, cand, m, k, seed, rng)
        else:
            pick = int(cand[np.argmax(logits[cand])])
        ids.append(pick)
        if pick == EOS_ID:
            break
        arr = np.array([[pick]], dtype=np.int64)
    return ids


def check_hash(ids: list[int], seed: str, m: int = 5) -> str:
    """Mean g-score (paper's Eq. 1, seed per layer = "{seed}_layer{n}") and
    z-score vs the expected 0.5, on any token span."""
    n = len(ids) * m
    if n == 0:
        return "Hash(0/0 frac=0.000 z=+0.0)"
    ones = sum(g_score(seed, layer, t) for t in ids for layer in range(m))
    frac = ones / n
    z = (frac - 0.5) / np.sqrt(0.25 / n)
    return f"Hash({ones}/{n} frac={frac:.3f} z={z:+.1f})"


def colored_hash(ids: list[int], seed: str, m: int = 5) -> str:
    """check_hash() colorized: green when the watermark is detected (z >= 1.65),
    red when it isn't."""
    n = len(ids) * m
    ones = sum(g_score(seed, layer, t) for t in ids for layer in range(m)) if ids else 0
    z = (ones / n - 0.5) / np.sqrt(0.25 / n) if n else 0.0
    return (Fore.GREEN if z >= 1.65 else Fore.RED) + check_hash(ids, seed, m)


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
        description="Invisible content hash via SynthID-Text-style tournament"
    )
    ap.add_argument("--prompt", type=str, default="What is gravity?")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--layers", type=int, default=5, metavar="M",
                    help="tournament layers / watermarking functions (default 5)")
    ap.add_argument("--competitors", type=int, default=2, metavar="K",
                    help="tokens per match; 2 is non-distortionary, >2 stronger (default 2)")
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--wrong-seed", default="negative key")
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = Model()
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
    kw = dict(
        max_new_tokens=args.tokens,
        model=model,
        vocab_size=tok.vocab_size,
        m=args.layers,
        k=args.competitors,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    wm = generate(prompt_ids, seed=args.seed, **kw)
    neg = generate(prompt_ids, seed=args.wrong_seed, **kw)
    plain = generate(prompt_ids, seed=None, **kw)
    wm, neg, plain = (ids[len(prompt_ids) :] for ids in (wm, neg, plain))

    print(
        Fore.CYAN
        + f"=== watermarked output ({args.layers} layers, {args.competitors} competitors/match) ==="
    )
    print(tok.decode(wm, skip_special_tokens=True))
    print(Fore.CYAN + "\n=== negative-seed output ===")
    print(tok.decode(neg, skip_special_tokens=True))
    print(Fore.CYAN + "\n=== plain output (baseline) ===")
    print(tok.decode(plain, skip_special_tokens=True))

    print(Fore.CYAN + "\n=== detection ===")
    print(f"{Fore.GREEN}watermarked, correct key  : {colored_hash(wm, args.seed, args.layers)}")
    print(f"{Fore.RED}watermarked, wrong key    : {colored_hash(wm, args.wrong_seed, args.layers)}")
    print(f"{Fore.RED}negative-seed, correct key: {colored_hash(neg, args.seed, args.layers)}")
    print(f"{Fore.RED}plain, correct key        : {colored_hash(plain, args.seed, args.layers)}")

    for name, text in [
        ("watermarked", wm),
        ("negative-seed", neg),
        ("plain", plain),
    ]:
        print(f"{Fore.CYAN}\n=== hash across splits of {name} text ===")
        print(f"{'from tok':>8} | {'seed':<32} {'wrong seed':<32}")
        for st in split_starts(text, tok):
            span = text[st:]
            print(
                f"{st:>8} | "
                f"{colored_hash(span, args.seed, args.layers):<32} "
                f"{colored_hash(span, args.wrong_seed, args.layers):<32}"
            )


if __name__ == "__main__":
    main()
