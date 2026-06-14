"""
kivi_qwen2 — KIVI KV-cache quantization for Qwen2 / Qwen2.5 models.

Implements the KIVI asymmetric quantization scheme (Liu et al., ICML 2024) as a
drop-in HuggingFace DynamicCache replacement, with no custom CUDA compilation
required.  Works with any transformers >= 4.43 Qwen2 model on GPU.

    from kivi_qwen2 import KIVICache

    cache = KIVICache(nbits=4, group_size=32, residual_length=128)
    outputs = model(input_ids, past_key_values=cache, use_cache=True)
"""
from kivi_qwen2.cache import KIVICache  # noqa: F401

__version__ = "0.1.0"
__all__ = ["KIVICache"]
