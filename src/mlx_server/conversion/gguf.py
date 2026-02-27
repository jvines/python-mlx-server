"""
GGUF → MLX direct conversion.

Reads a local GGUF file with mx.load (no external GGUF library needed) and
saves weights in the HF/transformers convention that mlx_lm expects.

Supported GGUF architectures: llama, qwen2, qwen2moe, mistral, mistral3, gemma, gemma2
Tokenizer must be downloaded from HF — provide hf_tokenizer_repo.

Memory notes:
  Q8_0 GGUFs stay natively quantized in MLX's Q8 format — no intermediate bloat.
  K-quant GGUFs (Q4_K_M, Q5_K_S, etc.) are dequantized to float16 by mx.load.
    70B K-quant → float16 requires ~140 GB intermediate — use HF path for 70B.
    ≤35B K-quant → float16 fits in 128 GB and is re-quantized when quantize=True.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx

from .jobs import ConversionJob

logger = logging.getLogger(__name__)

# ── Architecture metadata ────────────────────────────────────────────────────

_ARCH_MODEL_TYPE: Dict[str, str] = {
    "llama":    "llama",
    "qwen2":    "qwen2",
    "qwen2moe": "qwen2_moe",
    "mistral":  "mistral",
    "mistral3": "ministral3",
    "gemma":    "gemma",
    "gemma2":   "gemma2",
    "granite":  "granite",
}

_ARCH_CLASS: Dict[str, str] = {
    "llama":       "LlamaForCausalLM",
    "qwen2":       "Qwen2ForCausalLM",
    "qwen2_moe":   "Qwen2MoeForCausalLM",
    "mistral":     "MistralForCausalLM",
    "ministral3":  "MistralForCausalLM",
    "gemma":       "GemmaForCausalLM",
    "gemma2":      "Gemma2ForCausalLM",
    "granite":     "GraniteForCausalLM",
}

# GGUF metadata key → HF config.json field
# {arch} is substituted with the actual architecture string
_META_MAP: List[Tuple[str, str]] = [
    ("{arch}.embedding_length",                 "hidden_size"),
    ("{arch}.block_count",                      "num_hidden_layers"),
    ("{arch}.feed_forward_length",              "intermediate_size"),
    ("{arch}.attention.head_count",             "num_attention_heads"),
    ("{arch}.attention.head_count_kv",          "num_key_value_heads"),
    ("{arch}.attention.key_length",             "head_dim"),
    ("{arch}.context_length",                   "max_position_embeddings"),
    ("{arch}.rope.freq_base",                   "rope_theta"),
    ("{arch}.vocab_size",                       "vocab_size"),
    ("{arch}.attention.layer_norm_rms_epsilon", "rms_norm_eps"),
    ("{arch}.expert_count",                     "num_experts"),
    ("{arch}.expert_used_count",                "num_experts_per_tok"),
]

# ── Per-layer weight name map ────────────────────────────────────────────────

# GGUF per-block suffix → HF sub-path (same for llama, qwen2, mistral)
_LAYER_MAP: Dict[str, str] = {
    "attn_norm.weight":       "input_layernorm.weight",
    "ffn_norm.weight":        "post_attention_layernorm.weight",
    # Attention projections
    "attn_q.weight":          "self_attn.q_proj.weight",
    "attn_q.scales":          "self_attn.q_proj.scales",
    "attn_q.biases":          "self_attn.q_proj.biases",
    "attn_q.bias":            "self_attn.q_proj.bias",
    "attn_k.weight":          "self_attn.k_proj.weight",
    "attn_k.scales":          "self_attn.k_proj.scales",
    "attn_k.biases":          "self_attn.k_proj.biases",
    "attn_k.bias":            "self_attn.k_proj.bias",
    "attn_v.weight":          "self_attn.v_proj.weight",
    "attn_v.scales":          "self_attn.v_proj.scales",
    "attn_v.biases":          "self_attn.v_proj.biases",
    "attn_v.bias":            "self_attn.v_proj.bias",
    "attn_output.weight":     "self_attn.o_proj.weight",
    "attn_output.scales":     "self_attn.o_proj.scales",
    "attn_output.biases":     "self_attn.o_proj.biases",
    # QK norms (Llama 3.x, some Qwen3)
    "attn_q_norm.weight":     "self_attn.q_norm.weight",
    "attn_k_norm.weight":     "self_attn.k_norm.weight",
    # Dense FFN
    "ffn_gate.weight":        "mlp.gate_proj.weight",
    "ffn_gate.scales":        "mlp.gate_proj.scales",
    "ffn_gate.biases":        "mlp.gate_proj.biases",
    "ffn_up.weight":          "mlp.up_proj.weight",
    "ffn_up.scales":          "mlp.up_proj.scales",
    "ffn_up.biases":          "mlp.up_proj.biases",
    "ffn_down.weight":        "mlp.down_proj.weight",
    "ffn_down.scales":        "mlp.down_proj.scales",
    "ffn_down.biases":        "mlp.down_proj.biases",
    # MoE router
    "ffn_gate_inp.weight":    "mlp.gate.weight",
    "ffn_gate_inp.scales":    "mlp.gate.scales",
    "ffn_gate_inp.biases":    "mlp.gate.biases",
    # MoE shared expert (qwen2moe)
    "ffn_gate_shexp.weight":  "mlp.shared_expert.gate_proj.weight",
    "ffn_gate_shexp.scales":  "mlp.shared_expert.gate_proj.scales",
    "ffn_gate_shexp.biases":  "mlp.shared_expert.gate_proj.biases",
    "ffn_up_shexp.weight":    "mlp.shared_expert.up_proj.weight",
    "ffn_up_shexp.scales":    "mlp.shared_expert.up_proj.scales",
    "ffn_up_shexp.biases":    "mlp.shared_expert.up_proj.biases",
    "ffn_down_shexp.weight":  "mlp.shared_expert.down_proj.weight",
    "ffn_down_shexp.scales":  "mlp.shared_expert.down_proj.scales",
    "ffn_down_shexp.biases":  "mlp.shared_expert.down_proj.biases",
    # Shared expert gate (qwen2moe)
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
}

# MoE expert tensors stored as (n_experts, d_ff, d_model) in GGUF
_MOE_EXPS_MAP: Dict[str, str] = {
    "ffn_gate_exps": "gate_proj",
    "ffn_up_exps":   "up_proj",
    "ffn_down_exps": "down_proj",
}

_MOE_EXPS_RE = re.compile(
    r"^blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.(weight|scales|biases)$"
)
_LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "generation_config.json",
]


# ── Core helpers ─────────────────────────────────────────────────────────────


def _tick_progress(job: "ConversionJob", base_msg: str, interval: float = 5.0) -> Callable[[], None]:
    """
    Start a background thread that appends elapsed time to job.progress every
    `interval` seconds. Returns a stop() callable.
    """
    stop_event = threading.Event()
    start = time.monotonic()

    def _run() -> None:
        while not stop_event.wait(interval):
            elapsed = int(time.monotonic() - start)
            job.progress = f"{base_msg} ({elapsed}s elapsed)"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return stop_event.set


def _to_python(val: Any) -> Any:
    """Convert MLX / numpy scalars to plain Python types for JSON serialisation."""
    if hasattr(val, "item"):   # mx.array or numpy scalar
        return val.item()
    if hasattr(val, "tolist"):  # numpy array
        return val.tolist()
    return val


def _detect_quantization(weights: Dict[str, mx.array]) -> Optional[Dict[str, int]]:
    """
    Infer MLX quantization params from weight/scales tensor shapes.

    MLX packs quantized weights into uint32:
      bits=8 → 4 uint8 per uint32  → packed_in = original_in // 4
      bits=4 → 8 nibbles per uint32 → packed_in = original_in // 8

    group_size = original_in / n_scale_groups
    """
    for key, w in weights.items():
        if not key.endswith(".weight") or w.ndim != 2:
            continue
        scales_key = key[:-len(".weight")] + ".scales"
        s = weights.get(scales_key)
        if s is None or s.ndim != 2:
            continue
        packed_in, n_groups = w.shape[1], s.shape[1]
        for bits, pack_factor in ((8, 4), (4, 8)):
            original_in = packed_in * pack_factor
            if n_groups and original_in % n_groups == 0:
                group_size = original_in // n_groups
                if group_size in (16, 32, 64, 128):
                    return {"bits": bits, "group_size": group_size}
        break  # only need one layer to determine params
    return None


def _build_config(arch: str, meta: Dict[str, Any], weights: Dict[str, mx.array]) -> Dict[str, Any]:
    model_type = _ARCH_MODEL_TYPE[arch]

    def _get(template: str):
        val = meta.get(template.replace("{arch}", arch))
        return _to_python(val) if val is not None else None

    config: Dict[str, Any] = {
        "model_type": model_type,
        "architectures": [_ARCH_CLASS[model_type]],
        "torch_dtype": "bfloat16",
    }

    for gguf_tmpl, hf_key in _META_MAP:
        val = _get(gguf_tmpl)
        if val is not None:
            config[hf_key] = val

    # Fallback: infer vocab_size from embedding weight (already remapped)
    if "vocab_size" not in config:
        emb = weights.get("model.embed_tokens.weight")
        if emb is not None:
            config["vocab_size"] = int(emb.shape[0])
    # Second fallback: tokenizer vocabulary size stored in GGUF metadata
    if "vocab_size" not in config:
        n = _to_python(meta.get("tokenizer.ggml.token_count"))
        if n is not None:
            config["vocab_size"] = int(n)

    # Quantization config — tells mlx_lm to build QuantizedLinear layers
    q_info = _detect_quantization(weights)
    if q_info:
        config["quantization"] = q_info

    # Architecture-specific defaults
    if arch == "qwen2":
        config.setdefault("hidden_act", "silu")
        config.setdefault("attention_bias", True)
        config.setdefault("tie_word_embeddings", False)
    elif arch == "qwen2moe":
        config.setdefault("hidden_act", "silu")
        config.setdefault("attention_bias", True)
        config.setdefault("tie_word_embeddings", False)
    elif arch in ("llama", "mistral"):
        config.setdefault("hidden_act", "silu")
        config.setdefault("tie_word_embeddings", False)
    elif arch == "mistral3":
        config.setdefault("hidden_act", "silu")
        config["tie_word_embeddings"] = False
        # ministral3.py requires rope_parameters dict (not top-level rope_theta)
        rope_params: Dict[str, Any] = {}
        rope_theta = config.pop("rope_theta", None)
        if rope_theta is not None:
            rope_params["rope_theta"] = rope_theta
        # Try GGUF rope scaling keys (standard and variant spellings)
        rope_type = _get("mistral3.rope.scaling.type") or _get("mistral3.rope.scale_type")
        if rope_type:
            rope_params["type"] = rope_type
        rope_factor = _get("mistral3.rope.scaling.factor") or _get("mistral3.rope.scale")
        if rope_factor is not None:
            rope_params["factor"] = rope_factor
        orig_ctx = (
            _get("mistral3.rope.scaling.original_context_length")
            or _get("mistral3.context_length")
        )
        if orig_ctx is not None:
            rope_params["original_max_position_embeddings"] = int(orig_ctx)
        beta_fast = _get("mistral3.rope.scaling.beta_fast")
        if beta_fast is not None:
            rope_params["beta_fast"] = beta_fast
        beta_slow = _get("mistral3.rope.scaling.beta_slow")
        if beta_slow is not None:
            rope_params["beta_slow"] = beta_slow
        temp_scale = _get("mistral3.attention.temperature_scale")
        if temp_scale is not None:
            rope_params["llama_4_scaling_beta"] = temp_scale
        if rope_params:
            config["rope_parameters"] = rope_params
    elif arch == "granite":
        config.setdefault("attention_bias", False)
        config.setdefault("mlp_bias", False)
        config.setdefault("tie_word_embeddings", False)
        # Granite-specific scaling multipliers
        v = _get("granite.logit_scale")
        if v is not None:
            config["logits_scaling"] = v
        v = _get("granite.attention.scale")
        if v is not None:
            config["attention_multiplier"] = v
        v = _get("granite.embedding_scale")
        if v is not None:
            config["embedding_multiplier"] = v
        v = _get("granite.residual_scale")
        if v is not None:
            config["residual_multiplier"] = v

    return config


def _remap_weights(
    weights: Dict[str, mx.array], arch: str
) -> Dict[str, mx.array]:
    """Remap all GGUF weight keys to HF/transformers naming convention."""
    remapped: Dict[str, mx.array] = {}

    for key, array in weights.items():
        # Global weights
        if key in ("token_embd.weight", "token_embd.scales", "token_embd.biases"):
            suffix = key[len("token_embd"):]
            remapped[f"model.embed_tokens{suffix}"] = array
            continue
        if key == "output_norm.weight":
            remapped["model.norm.weight"] = array
            continue
        if key in ("output.weight", "output.scales", "output.biases"):
            suffix = key[len("output"):]
            remapped[f"lm_head{suffix}"] = array
            continue

        # MoE combined expert tensors — defer to split pass
        if _MOE_EXPS_RE.match(key):
            continue

        # Per-layer weights
        m = _LAYER_RE.match(key)
        if not m:
            logger.debug("Skipping unknown GGUF key: %s", key)
            continue

        layer_idx, suffix = m.group(1), m.group(2)
        hf_suffix = _LAYER_MAP.get(suffix)
        if hf_suffix is None:
            logger.debug("Skipping unmapped layer key: %s", key)
            continue

        remapped[f"model.layers.{layer_idx}.{hf_suffix}"] = array

    # Split MoE expert tensors
    if arch == "qwen2moe":
        remapped.update(_split_moe_experts(weights))

    return remapped


def _split_moe_experts(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Split combined MoE expert tensors (n_experts, d_ff, d_model) into per-expert weights."""
    result: Dict[str, mx.array] = {}
    for key, tensor in weights.items():
        m = _MOE_EXPS_RE.match(key)
        if not m:
            continue
        layer_idx = m.group(1)
        exps_name = m.group(2)  # ffn_gate_exps / ffn_up_exps / ffn_down_exps
        attr = m.group(3)       # weight / scales / biases
        proj = _MOE_EXPS_MAP[exps_name]

        n_experts = tensor.shape[0]
        for j in range(n_experts):
            hf_key = f"model.layers.{layer_idx}.mlp.experts.{j}.{proj}.{attr}"
            result[hf_key] = tensor[j]

    return result


def _is_already_quantized(weights: Dict[str, mx.array]) -> bool:
    return any(k.endswith(".scales") for k in weights)


def _dequantize_lm_head(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """
    Dequantize lm_head back to fp16 if it was quantized.

    Some architectures (Mistral, Mistral3) use an unquantized nn.Linear for
    lm_head and mlx_lm will reject lm_head.scales / lm_head.biases with
    strict=True. Dequantizing here is safe — mlx_lm won't re-quantize it on
    load, so there is a small memory cost for those layers at inference time.
    """
    if "lm_head.scales" not in weights:
        return weights
    w = weights["lm_head.weight"]
    scales = weights["lm_head.scales"]
    biases = weights["lm_head.biases"]
    q_info = _detect_quantization({"lm_head.weight": w, "lm_head.scales": scales})
    if q_info is None:
        return weights
    result = dict(weights)
    result["lm_head.weight"] = mx.dequantize(
        w, scales, biases, q_info["group_size"], q_info["bits"]
    )
    del result["lm_head.scales"]
    del result["lm_head.biases"]
    logger.info("gguf_remap: dequantized lm_head (architecture uses unquantized output projection)")
    return result


def _quantize_weights(
    weights: Dict[str, mx.array], q_bits: int, q_group_size: int
) -> Dict[str, mx.array]:
    """Apply MLX quantization to all 2-D linear weight matrices (in-place replacement)."""
    result: Dict[str, mx.array] = {}
    for key, array in weights.items():
        if key.endswith(".weight") and array.ndim == 2:
            q, scales, biases = mx.quantize(array, group_size=q_group_size, bits=q_bits)
            result[key] = q
            result[key[:-len(".weight")] + ".scales"] = scales
            result[key[:-len(".weight")] + ".biases"] = biases
        else:
            result[key] = array
    return result


def _save_weights(
    output_dir: Path,
    weights: Dict[str, mx.array],
    job: "ConversionJob",
    max_shard_gb: float = 4.0,
) -> None:
    """Save weights as safetensors, sharding if total size exceeds max_shard_gb."""
    # Evaluate all arrays (materialise lazy computations)
    mx.eval(list(weights.values()))

    total_bytes = sum(a.nbytes for a in weights.values())
    total_gb = total_bytes / 1024 ** 3
    max_bytes = int(max_shard_gb * 1024 ** 3)

    if total_bytes <= max_bytes:
        msg = f"Saving weights ({total_gb:.1f} GB)"
        job.progress = msg
        logger.info(msg)
        stop_tick = _tick_progress(job, msg)
        try:
            mx.save_safetensors(str(output_dir / "model.safetensors"), weights)
        finally:
            stop_tick()
        return

    # Shard
    shards: List[Dict[str, mx.array]] = []
    current: Dict[str, mx.array] = {}
    current_size = 0

    for key, array in weights.items():
        nb = array.nbytes
        if current and current_size + nb > max_bytes:
            shards.append(current)
            current = {}
            current_size = 0
        current[key] = array
        current_size += nb

    if current:
        shards.append(current)

    n = len(shards)
    weight_map: Dict[str, str] = {}
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        shard_gb = sum(a.nbytes for a in shard.values()) / 1024 ** 3
        msg = f"Saving shard {i + 1}/{n} ({shard_gb:.1f} GB)"
        job.progress = msg
        logger.info(msg)
        stop_tick = _tick_progress(job, msg)
        try:
            mx.save_safetensors(str(output_dir / fname), shard)
        finally:
            stop_tick()
        logger.info("Saved shard %d/%d", i + 1, n)
        for k in shard:
            weight_map[k] = fname

    index = {
        "metadata": {"total_size": str(total_bytes)},
        "weight_map": weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2)
    )


def _extract_tokenizer_from_gguf(metadata: Dict[str, Any], output_dir: Path) -> bool:
    """
    Reconstruct a HuggingFace-compatible tokenizer from GGUF metadata.

    Supports BPE (tokenizer.ggml.model == "gpt2") which covers Mistral, Qwen2,
    Llama 3, and most modern models. Returns True on success, False if the
    tokenizer type is unsupported (caller should fall back to hf_tokenizer_repo).
    """
    tok_model = str(metadata.get("tokenizer.ggml.model", "")).lower()
    tokens_raw = metadata.get("tokenizer.ggml.tokens")
    if tokens_raw is None:
        return False

    tokens: List[str] = (
        [str(t) for t in tokens_raw.tolist()]
        if hasattr(tokens_raw, "tolist")
        else [str(t) for t in tokens_raw]
    )

    merges_raw = metadata.get("tokenizer.ggml.merges")
    has_merges = merges_raw is not None and len(merges_raw) > 0

    if tok_model not in ("gpt2", "") or not has_merges:
        logger.info("gguf_tokenizer: model type '%s' not supported for extraction", tok_model)
        return False

    merges: List[str] = (
        [str(m) for m in merges_raw.tolist()]
        if hasattr(merges_raw, "tolist")
        else [str(m) for m in merges_raw]
    )

    # Token types: 0=normal, 1=byte, 3=control, 4=user_defined, 6=unused
    token_types_raw = metadata.get("tokenizer.ggml.token_type")
    token_types: Optional[List[int]] = None
    if token_types_raw is not None:
        token_types = (
            token_types_raw.tolist()
            if hasattr(token_types_raw, "tolist")
            else list(token_types_raw)
        )

    def _tok_id(key: str) -> Optional[int]:
        v = _to_python(metadata.get(key))
        return int(v) if v is not None else None

    bos_id  = _tok_id("tokenizer.ggml.bos_token_id")
    eos_id  = _tok_id("tokenizer.ggml.eos_token_id")
    unk_id  = _tok_id("tokenizer.ggml.unknown_token_id") or _tok_id("tokenizer.ggml.unk_token_id")

    def _tok_str(tid: Optional[int]) -> Optional[str]:
        return tokens[tid] if tid is not None and 0 <= tid < len(tokens) else None

    bos_token = _tok_str(bos_id)
    eos_token = _tok_str(eos_id)
    unk_token = _tok_str(unk_id)

    # Special tokens: control (3) or user-defined (4) type
    SPECIAL_TYPES = {3, 4}
    added_tokens = []
    for i, tok in enumerate(tokens):
        tt = token_types[i] if token_types else 0
        if tt in SPECIAL_TYPES:
            added_tokens.append({
                "id": i, "content": tok, "single_word": False,
                "lstrip": False, "rstrip": False,
                "normalized": False, "special": True,
            })

    vocab = {tok: i for i, tok in enumerate(tokens)}

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel", "add_prefix_space": False,
            "trim_offsets": True, "use_regex": True,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel", "add_prefix_space": False,
            "trim_offsets": True, "use_regex": True,
        },
        "model": {
            "type": "BPE", "dropout": None, "unk_token": unk_token,
            "continuing_subword_prefix": None, "end_of_word_suffix": None,
            "fuse_unk": False, "byte_fallback": False,
            "vocab": vocab, "merges": merges,
        },
    }
    (output_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer_json, indent=2, ensure_ascii=False)
    )

    add_bos = bool(_to_python(metadata.get("tokenizer.ggml.add_bos_token", True)))
    add_eos = bool(_to_python(metadata.get("tokenizer.ggml.add_eos_token", False)))
    chat_template = metadata.get("tokenizer.chat_template")

    tokenizer_config: Dict[str, Any] = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 131072,
        "bos_token": bos_token,
        "eos_token": eos_token,
        "unk_token": unk_token,
        "pad_token": unk_token or eos_token,
        "add_bos_token": add_bos,
        "add_eos_token": add_eos,
    }
    if chat_template:
        tokenizer_config["chat_template"] = str(chat_template)
    (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))

    special_tokens_map: Dict[str, Any] = {}
    if bos_token:
        special_tokens_map["bos_token"] = bos_token
    if eos_token:
        special_tokens_map["eos_token"] = eos_token
    if unk_token:
        special_tokens_map["unk_token"] = unk_token
    (output_dir / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2))

    logger.info(
        "gguf_tokenizer: extracted from GGUF  vocab=%d  merges=%d  special=%d",
        len(tokens), len(merges), len(added_tokens),
    )
    return True


def _download_tokenizer(hf_repo: str, output_dir: Path) -> List[str]:
    from huggingface_hub import hf_hub_download

    copied: List[str] = []
    for filename in _TOKENIZER_FILES:
        try:
            cached = hf_hub_download(repo_id=hf_repo, filename=filename, token=False)
            shutil.copy2(cached, output_dir / filename)
            logger.info("gguf_tokenizer: %s", filename)
            copied.append(filename)
        except Exception as exc:
            logger.debug(
                "gguf_tokenizer: skipped %s from %s (%s)",
                filename,
                hf_repo,
                exc,
            )

    # At least one tokenizer payload is required for mlx_lm.load(...)
    if not any(f in copied for f in ("tokenizer.json", "tokenizer.model")):
        raise RuntimeError(
            f"Failed to download tokenizer payload from '{hf_repo}'. "
            "Expected tokenizer.json or tokenizer.model."
        )
    return copied


# ── Public entry point ───────────────────────────────────────────────────────


def convert_from_gguf(
    job: ConversionJob,
    gguf_path: str,
    hf_tokenizer_repo: Optional[str] = None,
    prefer_hf_tokenizer: bool = False,
    quantize: bool = True,
    q_bits: int = 4,
    q_group_size: int = 64,
) -> None:
    """
    Convert a local GGUF file to an MLX-compatible model directory.

    Args:
        job:               Conversion job for progress tracking.
        gguf_path:         Absolute path to the .gguf file.
        hf_tokenizer_repo: HuggingFace repo ID to download tokenizer from
                           (e.g. "mistralai/Ministral-3B-Instruct-2410").
                           Optional fallback when GGUF tokenizer extraction fails.
        prefer_hf_tokenizer:
                           If True and hf_tokenizer_repo is set, skip GGUF
                           tokenizer extraction and force HF tokenizer files.
        quantize:          Re-quantize fp16 weights (applies to K-quant GGUFs).
                           Q8_0 GGUFs are already quantized — this flag is ignored.
        q_bits:            Target quantisation bits (4 or 8).
        q_group_size:      Quantisation group size.
    """
    output = Path(job.output_path)
    if output.exists():
        raise FileExistsError(
            f"Output path already exists: {output}. "
            "Delete it or choose a different path."
        )
    output.mkdir(parents=True)

    # ── 1. Load GGUF ─────────────────────────────────────────────────────────
    gguf_path_obj = Path(gguf_path)
    gguf_size_gb = gguf_path_obj.stat().st_size / 1024 ** 3

    logger.info("gguf_load: path    = %s", gguf_path)
    logger.info("gguf_load: size    = %.2f GB", gguf_size_gb)

    load_msg = f"Loading GGUF ({gguf_size_gb:.1f} GB)"
    job.progress = load_msg
    stop_tick = _tick_progress(job, load_msg)
    t_load = time.monotonic()
    try:
        weights, metadata = mx.load(gguf_path, return_metadata=True)
    finally:
        stop_tick()
    t_load = time.monotonic() - t_load
    weights = dict(weights)

    arch = metadata.get("general.architecture", "")
    if arch not in _ARCH_MODEL_TYPE:
        raise ValueError(
            f"Unsupported GGUF architecture: {arch!r}. "
            f"Supported: {list(_ARCH_MODEL_TYPE)}"
        )

    model_name = metadata.get("general.name", "") or gguf_path_obj.stem
    logger.info("gguf_load: loaded  = %.1fs", t_load)
    logger.info("gguf_load: arch    = %s", arch)
    logger.info("gguf_load: name    = %s", model_name)
    logger.info("gguf_load: tensors = %d", len(weights))

    # Print key model dimensions
    def _meta(key: str):
        v = metadata.get(key.replace("{arch}", arch))
        return _to_python(v) if v is not None else "?"

    logger.info("gguf_meta: hidden_size             = %s", _meta("{arch}.embedding_length"))
    logger.info("gguf_meta: num_hidden_layers        = %s", _meta("{arch}.block_count"))
    logger.info("gguf_meta: num_attention_heads      = %s", _meta("{arch}.attention.head_count"))
    logger.info("gguf_meta: num_key_value_heads      = %s", _meta("{arch}.attention.head_count_kv"))
    logger.info("gguf_meta: head_dim                 = %s", _meta("{arch}.attention.key_length"))
    logger.info("gguf_meta: intermediate_size        = %s", _meta("{arch}.feed_forward_length"))
    logger.info("gguf_meta: max_position_embeddings  = %s", _meta("{arch}.context_length"))
    logger.info("gguf_meta: rope_theta               = %s", _meta("{arch}.rope.freq_base"))
    logger.info("gguf_meta: vocab_size               = %s", _meta("{arch}.vocab_size"))
    logger.info("gguf_meta: rms_norm_eps             = %s", _meta("{arch}.attention.layer_norm_rms_epsilon"))

    # ── 2. Remap weight names ────────────────────────────────────────────────
    job.progress = "Remapping weight names"
    logger.info("gguf_remap: remapping %d tensors ...", len(weights))
    t_remap = time.monotonic()
    remapped = _remap_weights(weights, arch)
    t_remap = time.monotonic() - t_remap
    n_skipped = len(weights) - len(remapped)
    del weights
    logger.info("gguf_remap: %d → %d tensors  (%d skipped)  %.2fs",
                len(remapped) + n_skipped, len(remapped), n_skipped, t_remap)

    # ── 3. Optionally re-quantize ────────────────────────────────────────────
    already_q = _is_already_quantized(remapped)
    if already_q:
        q_info = _detect_quantization(remapped)
        bits_str = f"Q{q_info['bits']}_0" if q_info else "quantized"
        job.progress = f"Weights already {bits_str} — skipping re-quantization"
        logger.info("gguf_quant: already %s (group_size=%s) — skipping",
                    bits_str, q_info["group_size"] if q_info else "?")
    elif quantize:
        job.progress = f"Quantizing to {q_bits}-bit (group_size={q_group_size})"
        logger.info("gguf_quant: quantizing to %d-bit, group_size=%d ...", q_bits, q_group_size)
        t_q = time.monotonic()
        remapped = _quantize_weights(remapped, q_bits, q_group_size)
        logger.info("gguf_quant: done  %.1fs", time.monotonic() - t_q)
    else:
        logger.info("gguf_quant: quantize=False, keeping fp16")

    # ── 4. Build config.json ─────────────────────────────────────────────────
    job.progress = "Writing config.json"
    config = _build_config(arch, metadata, remapped)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    logger.info("gguf_config: model_type=%s  vocab_size=%s  quantization=%s",
                config.get("model_type"), config.get("vocab_size"),
                config.get("quantization", "none"))

    # ── 5. Tokenizer ──────────────────────────────────────────────────────────
    job.progress = "Preparing tokenizer"
    if prefer_hf_tokenizer and hf_tokenizer_repo:
        job.progress = f"Downloading tokenizer from {hf_tokenizer_repo}"
        logger.info("gguf_tokenizer: fetching from %s", hf_tokenizer_repo)
        copied = _download_tokenizer(hf_tokenizer_repo, output)
        logger.info("gguf_tokenizer: downloaded %d files", len(copied))
    elif _extract_tokenizer_from_gguf(metadata, output):
        logger.info("gguf_tokenizer: extracted from GGUF metadata")
    elif hf_tokenizer_repo:
        job.progress = f"Downloading tokenizer from {hf_tokenizer_repo}"
        logger.info("gguf_tokenizer: extraction failed; fetching from %s", hf_tokenizer_repo)
        copied = _download_tokenizer(hf_tokenizer_repo, output)
        logger.info("gguf_tokenizer: downloaded %d files", len(copied))
    else:
        raise RuntimeError(
            "Could not extract tokenizer from GGUF metadata and no "
            "hf_tokenizer_repo was provided. Re-submit with hf_tokenizer_repo set."
        )

    # ── 6. Save weights ───────────────────────────────────────────────────────
    _save_weights(output, remapped, job)

    job.progress = "Done"
    logger.info("gguf_convert: complete → %s", output)
