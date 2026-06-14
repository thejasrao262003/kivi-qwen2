# kivi_qwen2

KIVI asymmetric KV-cache quantization (Liu et al., ICML 2024) as a plug-and-play
HuggingFace `DynamicCache` drop-in for **Qwen2 / Qwen2.5** models.

The original KIVI repo supports only Llama and Mistral.  This package ports the
quantisation scheme to Qwen2's attention interface (transformers ≥ 4.43) with no
custom CUDA build step required — just `pip install`.

## What is KIVI?

KIVI quantises the KV cache asymmetrically:

| Cache | Quantisation axis | Rationale |
|-------|-------------------|-----------|
| Keys  | Per-channel (groups of `group_size` *tokens* share a scale per head-dim channel) | Key distributions vary more across channels |
| Values | Per-token (groups of `group_size` *features* share a scale per sequence token) | Value distributions vary more across tokens |

A rolling FP16 residual buffer (`residual_length` most-recent tokens) is kept
unquantised for recency fidelity.

## Differences from the original KIVI repo

| | jy-yuan/KIVI | kivi_qwen2 |
|---|---|---|
| Models | Llama-2/3, Mistral | Qwen2, Qwen2.5 |
| Attention kernel | Custom fused bmm (CUDA) | Standard attention (dequant→FP16→attn) |
| Build step | Requires `nvcc` | Pure Python + PyTorch, no build |
| Cache API | Custom tuple | HF `DynamicCache` drop-in |

Quality (perplexity) is identical — the fused kernel is a latency optimisation
only; the quantisation arithmetic is the same.

## Installation

```bash
pip install torch>=2.0 transformers>=4.43 accelerate
pip install .           # from this directory
# or
pip install git+https://github.com/<your-fork>/kivi_qwen2.git
```

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kivi_qwen2 import KIVICache

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [{"role": "user", "content": "Explain KV cache quantisation in 3 sentences."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# INT4 KV cache with KIVI quantisation
cache = KIVICache(nbits=4, group_size=32, residual_length=128)

with torch.inference_mode():
    # Prefill
    outputs = model(input_ids, past_key_values=cache, use_cache=True)
    past = outputs.past_key_values

    # Decode
    generated = []
    next_logits = outputs.logits[:, -1, :]
    for _ in range(256):
        next_id = next_logits.argmax(-1, keepdim=True)
        if next_id.item() in tokenizer.eos_token_id:
            break
        generated.append(next_id.item())
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        next_logits = out.logits[:, -1, :]
        past = out.past_key_values

print(tokenizer.decode(generated, skip_special_tokens=True))
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nbits` | `4` | Bit-width: 2, 4, or 8 |
| `group_size` | `32` | Tokens per key-group; features per value-group. Must divide `head_dim`. Qwen2.5-7B: `head_dim=128`, so 32 works. |
| `residual_length` | `128` | Recent tokens kept in full FP16 precision |

## Memory footprint (Qwen2.5-7B, 28K context)

| Cache format | KV memory |
|---|---|
| FP16 (baseline) | ~1.6 GB across 28 layers |
| KIVI INT4 | ~200 MB across 28 layers (~8× reduction) |
| KIVI INT2 | ~110 MB across 28 layers (~14× reduction) |

## Citation

If you use this in your work, please also cite the original KIVI paper:

```bibtex
@inproceedings{liu2024kivi,
  title     = {{KIVI}: A Tuning-Free Asymmetric 2bit Quantization for {KV} Cache},
  author    = {Liu, Zirui and Yuan, Jiayi and Jin, Hongye and Zhong, Shaochen
               and Xu, Zhaozhuo and Braverman, Vladimir and Chen, Beidi and Hu, Xia},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024},
}
```

## License

MIT — same as the original KIVI repository.
