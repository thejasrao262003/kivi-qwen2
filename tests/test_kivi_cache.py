"""
CPU round-trip tests for KIVICache.

These tests run without a GPU by operating on CPU float32 tensors.
They verify quantization arithmetic and cache state management, not
GPU performance or model integration.
"""
import torch
import pytest
from kivi_qwen2 import KIVICache
from kivi_qwen2.quant_utils import (
    quant_and_pack_kcache,
    quant_and_pack_vcache,
    unpack_and_dequant_kcache,
    unpack_and_dequant_vcache,
)

B, NH, D = 1, 4, 64  # batch, heads, head_dim
GROUP_SIZE = 32


# ---------------------------------------------------------------------------
# quant_utils round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [4, 8])
def test_kcache_roundtrip(bits):
    T = GROUP_SIZE * 4
    k = torch.randn(B, NH, T, D)
    code, scale, mn = quant_and_pack_kcache(k, GROUP_SIZE, bits)
    k_hat = unpack_and_dequant_kcache(code, scale, mn, GROUP_SIZE, bits)
    assert k_hat.shape == k.shape
    rel_err = (k - k_hat).abs().mean() / k.abs().mean().clamp_min(1e-6)
    max_err = {4: 0.12, 8: 0.005}[bits]
    assert rel_err < max_err, f"INT{bits} key round-trip error {rel_err:.4f} > {max_err}"


@pytest.mark.parametrize("bits", [4, 8])
def test_vcache_roundtrip(bits):
    T = 8
    v = torch.randn(B, NH, T, D)
    code, scale, mn = quant_and_pack_vcache(v, GROUP_SIZE, bits)
    v_hat = unpack_and_dequant_vcache(code, scale, mn, GROUP_SIZE, bits)
    assert v_hat.shape == v.shape
    rel_err = (v - v_hat).abs().mean() / v.abs().mean().clamp_min(1e-6)
    max_err = {4: 0.12, 8: 0.005}[bits]
    assert rel_err < max_err, f"INT{bits} value round-trip error {rel_err:.4f} > {max_err}"


# ---------------------------------------------------------------------------
# KIVICache state management
# ---------------------------------------------------------------------------

def _make_kv(T):
    return torch.randn(B, NH, T, D), torch.randn(B, NH, T, D)


def test_cache_accumulates_seq_length():
    cache = KIVICache(nbits=4, group_size=GROUP_SIZE, residual_length=64)
    total = 0
    for t in [16, 32, 16]:
        k, v = _make_kv(t)
        cache.update(k, v, layer_idx=0)
        total += t
        assert cache.get_seq_length(0) == total


def test_residual_only_while_under_limit():
    residual = 64
    cache = KIVICache(nbits=4, group_size=GROUP_SIZE, residual_length=residual)
    k, v = _make_kv(residual)
    cache.update(k, v, layer_idx=0)
    # Nothing should be quantised yet — all tokens in residual buffer
    assert 0 not in cache._k_quant
    assert cache.get_seq_length(0) == residual


def test_flush_triggers_on_overflow():
    residual = 64
    cache = KIVICache(nbits=4, group_size=GROUP_SIZE, residual_length=residual)
    # Fill residual then add enough to force a flush
    k, v = _make_kv(residual + GROUP_SIZE)
    cache.update(k, v, layer_idx=0)
    assert 0 in cache._k_quant, "Quantised storage should be populated after overflow"


def test_materialize_shape_matches_input():
    cache = KIVICache(nbits=4, group_size=GROUP_SIZE, residual_length=32)
    T_total = 0
    for t in [16, 32, 64, 16]:
        k, v = _make_kv(t)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        T_total += t
        assert k_out.shape == (B, NH, T_total, D)
        assert v_out.shape == (B, NH, T_total, D)


def test_multiple_layers_independent():
    cache = KIVICache(nbits=4, group_size=GROUP_SIZE, residual_length=32)
    for layer in range(4):
        k, v = _make_kv(GROUP_SIZE * 3)
        cache.update(k, v, layer_idx=layer)
    for layer in range(4):
        assert cache.get_seq_length(layer) == GROUP_SIZE * 3


def test_invalid_nbits_raises():
    with pytest.raises(ValueError):
        KIVICache(nbits=3)
