"""Plain greedy generation with nucleus sampling — no watermark.

Provides :class:`GreedyGenerator` which uses the same KV-cache loop and
nucleus filtering as :class:`WatermarkGenerator`, but always picks the
highest-logit candidate. Useful as a baseline or when you want generation
without any watermarking overhead.
"""

from target_hash_gen.core import GreedyGenerator

MODEL_ID = "LiquidAI/LFM2.5-350M"
EOS_ID = 7

__all__ = ["GreedyGenerator", "MODEL_ID", "EOS_ID"]
