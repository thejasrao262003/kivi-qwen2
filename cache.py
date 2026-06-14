"""
KIVICache — KIVI asymmetric KV-cache quantization as a HuggingFace DynamicCache.

Implements the quantization scheme from:
    Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
    ICML 2024  |  https://arxiv.org/abs/2402.02750

Design:
  - Extends transformers.DynamicCache; no model code changes required.
  - Overrides update() to quantize old tokens and dequantize at attention time.
  - Maintains a rolling FP16 residual buffer for the most recent `residual_length`
    tokens, matching KIVI's original rolling-window design.
  - Dequantizes to FP16 before returning K,V so the model's attention kernel
    runs unchanged (we skip KIVI's custom fused bmm kernel; quality is identical).

Compatible with any transformers >= 4.43 model that uses the Cache API
(DynamicCache, SDPA, FlashAttention2 all work as-is).

Example usage (Qwen2.5-7B-Instruct):

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kivi_qwen2 import KIVICache

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    prompt = tokenizer.apply_chat_template([{"role":"user","content":"Hello"}],
                                           tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # Drop-in replacement for the default DynamicCache:
    cache = KIVICache(nbits=4, group_size=32, residual_length=128)
    with torch.inference_mode():
        out = model(input_ids, past_key_values=cache, use_cache=True)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import DynamicCache

from kivi_qwen2.quant_utils import (
    quant_and_pack_kcache,
    quant_and_pack_vcache,
    unpack_and_dequant_kcache,
    unpack_and_dequant_vcache,
)


class KIVICache(DynamicCache):
    """KIVI asymmetric KV-cache quantization as a DynamicCache drop-in.

    Args:
        nbits:            Quantisation bit-width — 2, 4, or 8.
        group_size:       Tokens per quant group for keys; features per group for values.
                          Must divide head_dim (Qwen2.5-7B: head_dim=128, so 32 works).
        residual_length:  Number of most-recent tokens kept in full FP16.
    """

    def __init__(
        self,
        nbits: int = 4,
        group_size: int = 32,
        residual_length: int = 128,
    ) -> None:
        super().__init__()
        if nbits not in (2, 4, 8):
            raise ValueError(f"nbits must be 2, 4, or 8; got {nbits}")
        self.nbits = nbits
        self.group_size = group_size
        self.residual_length = residual_length

        # Per-layer quantised key storage (packed along T dim)
        self._k_quant: Dict[int, torch.Tensor] = {}
        self._k_scale: Dict[int, torch.Tensor] = {}
        self._k_mn:    Dict[int, torch.Tensor] = {}
        self._k_full:  Dict[int, torch.Tensor] = {}   # FP16 residual

        # Per-layer quantised value storage (packed along D dim)
        self._v_quant: Dict[int, torch.Tensor] = {}
        self._v_scale: Dict[int, torch.Tensor] = {}
        self._v_mn:    Dict[int, torch.Tensor] = {}
        self._v_full:  Dict[int, torch.Tensor] = {}   # FP16 residual

    # ------------------------------------------------------------------
    # DynamicCache interface
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K,V tokens; quantise overflow; return full dequantised K,V."""
        # Accumulate new tokens in FP16 residual buffer
        if layer_idx not in self._k_full:
            self._k_full[layer_idx] = key_states
            self._v_full[layer_idx] = value_states
        else:
            self._k_full[layer_idx] = torch.cat(
                [self._k_full[layer_idx], key_states], dim=2
            )
            self._v_full[layer_idx] = torch.cat(
                [self._v_full[layer_idx], value_states], dim=2
            )

        # Flush any complete quantisation groups that exceed residual_length
        full_len = self._k_full[layer_idx].shape[2]
        if full_len > self.residual_length:
            excess = full_len - self.residual_length
            # Only quantise complete groups to satisfy T % group_size == 0
            n_to_quant = (excess // self.group_size) * self.group_size
            if n_to_quant > 0:
                self._flush_to_quant(layer_idx, n_to_quant)

        return self._materialize_kv(layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Total sequence length (quantised + residual) seen at layer_idx."""
        feat_per_int = 32 // self.nbits
        q_len = (
            self._k_quant[layer_idx].shape[2] * feat_per_int
            if layer_idx in self._k_quant
            else 0
        )
        f_len = (
            self._k_full[layer_idx].shape[2] if layer_idx in self._k_full else 0
        )
        return q_len + f_len

    def get_max_cache_shape(self) -> Optional[Tuple[int, ...]]:
        return None   # dynamic; no fixed upper bound

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_to_quant(self, layer_idx: int, n: int) -> None:
        """Move the first n tokens of the FP16 residual into quantised storage."""
        keys_q = self._k_full[layer_idx][:, :, :n, :].contiguous()
        vals_q = self._v_full[layer_idx][:, :, :n, :].contiguous()
        self._k_full[layer_idx] = self._k_full[layer_idx][:, :, n:, :]
        self._v_full[layer_idx] = self._v_full[layer_idx][:, :, n:, :]

        kq, ks, km = quant_and_pack_kcache(keys_q, self.group_size, self.nbits)
        vq, vs, vm = quant_and_pack_vcache(vals_q, self.group_size, self.nbits)

        if layer_idx not in self._k_quant:
            self._k_quant[layer_idx] = kq
            self._k_scale[layer_idx] = ks
            self._k_mn[layer_idx]    = km
            self._v_quant[layer_idx] = vq
            self._v_scale[layer_idx] = vs
            self._v_mn[layer_idx]    = vm
        else:
            # Keys: code (B,nh,T_packed,D), scale (B,nh,num_groups,1,D) → cat dim=2
            self._k_quant[layer_idx] = torch.cat([self._k_quant[layer_idx], kq], dim=2)
            self._k_scale[layer_idx] = torch.cat([self._k_scale[layer_idx], ks], dim=2)
            self._k_mn[layer_idx]    = torch.cat([self._k_mn[layer_idx],    km], dim=2)
            # Values: code (B,nh,T,D_packed), scale (B,nh,T,num_groups,1) → cat dim=2
            self._v_quant[layer_idx] = torch.cat([self._v_quant[layer_idx], vq], dim=2)
            self._v_scale[layer_idx] = torch.cat([self._v_scale[layer_idx], vs], dim=2)
            self._v_mn[layer_idx]    = torch.cat([self._v_mn[layer_idx],    vm], dim=2)

    def _materialize_kv(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantise stored tokens and concatenate with FP16 residual."""
        k_res = self._k_full[layer_idx]
        v_res = self._v_full[layer_idx]

        if layer_idx not in self._k_quant:
            return k_res, v_res

        k_deq = unpack_and_dequant_kcache(
            self._k_quant[layer_idx],
            self._k_scale[layer_idx],
            self._k_mn[layer_idx],
            self.group_size,
            self.nbits,
        )
        v_deq = unpack_and_dequant_vcache(
            self._v_quant[layer_idx],
            self._v_scale[layer_idx],
            self._v_mn[layer_idx],
            self.group_size,
            self.nbits,
        )
        return (
            torch.cat([k_deq, k_res], dim=2),
            torch.cat([v_deq, v_res], dim=2),
        )
