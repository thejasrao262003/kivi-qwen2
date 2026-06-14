"""
Quantization helpers adapted from KIVI (Liu et al., ICML 2024).

Original: https://github.com/jy-yuan/KIVI/blob/main/quant/new_pack.py  (MIT)

Adaptation notes:
- Pure PyTorch + no custom CUDA extension; Triton kernels omitted.
- Adds `clamp_min_(1e-8)` on scale to guard against zero-range groups.
- Removes dependency on the `quant` C extension package so the module
  installs with a plain `pip install` on any CUDA-capable machine.

KIVI quantization asymmetry:
  Keys   → per-channel: groups of group_size tokens share a scale per head_dim channel.
  Values → per-token:   groups of group_size features share a scale per sequence token.
"""
import torch


def quant_and_pack_kcache(
    k: torch.Tensor,
    group_size: int,
    bits: int,
):
    """Quantize and bit-pack key cache along the sequence (T) dimension.

    Args:
        k:          (B, nh, T, D) FP16 key tensor; T must be divisible by group_size.
        group_size: Number of tokens per quantization group.
        bits:       Bit-width — 2, 4, or 8.

    Returns:
        code:  (B, nh, T // (32 // bits), D) packed int32
        scale: (B, nh, T // group_size, 1, D) per-group-per-channel scale
        mn:    (B, nh, T // group_size, 1, D) per-group-per-channel minimum
    """
    assert bits in (2, 4, 8), f"bits must be 2, 4 or 8; got {bits}"
    assert k.dim() == 4, "k must be 4-D (B, nh, T, D)"
    B, nh, T, D = k.shape
    assert T % group_size == 0, f"T={T} must be divisible by group_size={group_size}"
    max_int = (1 << bits) - 1
    num_groups = T // group_size

    data = k.view(B, nh, num_groups, group_size, D)
    mn = data.min(dim=3, keepdim=True)[0]            # (B, nh, num_groups, 1, D)
    mx = data.max(dim=3, keepdim=True)[0]
    scale = ((mx - mn) / max_int).clamp_min_(1e-8)
    data = ((data - mn) / scale).clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(B, nh, T, D)
    code = _pack_tensor(data, bits, pack_dim=2)
    return code, scale, mn


def quant_and_pack_vcache(
    v: torch.Tensor,
    group_size: int,
    bits: int,
):
    """Quantize and bit-pack value cache along the feature (D) dimension.

    Args:
        v:          (B, nh, T, D) FP16 value tensor; D must be divisible by group_size.
        group_size: Number of features per quantization group.
        bits:       Bit-width — 2, 4, or 8.

    Returns:
        code:  (B, nh, T, D // (32 // bits)) packed int32
        scale: (B, nh, T, D // group_size, 1) per-token-per-group scale
        mn:    (B, nh, T, D // group_size, 1) per-token-per-group minimum
    """
    assert bits in (2, 4, 8), f"bits must be 2, 4 or 8; got {bits}"
    assert v.dim() == 4, "v must be 4-D (B, nh, T, D)"
    B, nh, T, D = v.shape
    assert D % group_size == 0, f"D={D} must be divisible by group_size={group_size}"
    max_int = (1 << bits) - 1
    num_groups = D // group_size

    data = v.view(B, nh, T, num_groups, group_size)
    mn = data.min(dim=4, keepdim=True)[0]            # (B, nh, T, num_groups, 1)
    mx = data.max(dim=4, keepdim=True)[0]
    scale = ((mx - mn) / max_int).clamp_min_(1e-8)
    data = ((data - mn) / scale).clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(B, nh, T, D)
    code = _pack_tensor(data, bits, pack_dim=3)
    return code, scale, mn


def unpack_and_dequant_kcache(
    k_code: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    """Inverse of quant_and_pack_kcache.

    Returns:
        (B, nh, T, D) FP16 tensor.
    """
    data = _unpack_tensor(k_code, bits, pack_dim=2)   # (B, nh, T, D)  int16
    B, nh, T, D = data.shape
    num_groups = T // group_size
    data = data.view(B, nh, num_groups, group_size, D).to(torch.float16)
    data = data * scale + mn                           # scale/mn broadcast over group_size dim
    return data.view(B, nh, T, D)


def unpack_and_dequant_vcache(
    v_code: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    """Inverse of quant_and_pack_vcache.

    Returns:
        (B, nh, T, D) FP16 tensor.
    """
    data = _unpack_tensor(v_code, bits, pack_dim=3)   # (B, nh, T, D)  int16
    B, nh, T, D = data.shape
    num_groups = D // group_size
    data = data.view(B, nh, T, num_groups, group_size).to(torch.float16)
    data = data * scale + mn
    return data.view(B, nh, T, D)


# ---------------------------------------------------------------------------
# Internal bit-packing helpers (pure PyTorch, pack_dim ∈ {2, 3} only)
# ---------------------------------------------------------------------------

def _pack_tensor(data: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    """Pack multiple low-bit integers into int32 along pack_dim."""
    shape = data.shape
    feat_per_int = 32 // bits
    assert shape[pack_dim] % feat_per_int == 0
    out_shape = shape[:pack_dim] + (shape[pack_dim] // feat_per_int,) + shape[pack_dim + 1:]
    code = torch.zeros(out_shape, dtype=torch.int32, device=data.device)
    ui = [slice(None)] * len(shape)
    pi = [slice(None)] * len(shape)
    row, cursor = 0, 0
    while row < code.shape[pack_dim]:
        pi[pack_dim] = row
        for j in range(cursor, cursor + feat_per_int):
            ui[pack_dim] = j
            code[tuple(pi)] |= data[tuple(ui)] << (bits * (j - cursor))
        cursor += feat_per_int
        row += 1
    return code


def _unpack_tensor(v_code: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    """Unpack int32 tensor back to low-bit integers (vectorised, no Python loop)."""
    assert pack_dim in (2, 3), "pack_dim must be 2 or 3"
    shape = v_code.shape
    feat_per_int = 32 // bits
    new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim + 1:]
    mask = 0xFF >> (8 - bits)
    idx = torch.arange(new_shape[pack_dim], device=v_code.device)
    i_idx = idx // feat_per_int
    j_idx = idx % feat_per_int
    sel = [slice(None)] * len(new_shape)
    sel[pack_dim] = i_idx
    if pack_dim == 2:
        unpacked = (v_code[tuple(sel)] >> (j_idx * bits)[None, None, :, None]).to(torch.int16) & mask
    else:  # pack_dim == 3
        unpacked = (v_code[tuple(sel)] >> (j_idx * bits)).to(torch.int16) & mask
    return unpacked
