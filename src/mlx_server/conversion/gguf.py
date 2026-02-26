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
    "mistral3": "mistral",
    "gemma":    "gemma",
    "gemma2":   "gemma2",
}

_ARCH_CLASS: Dict[str, str] = {
    "llama":     "LlamaForCausalLM",
    "qwen2":     "Qwen2ForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
    "mistral":   "MistralForCausalLM",
    "mistral3":  "MistralForCausalLM",
    "gemma":     "GemmaForCausalLM",
    "gemma2":    "Gemma2ForCausalLM",
}

# GGUF metadata key → HF config.json field
# {arch} is substituted with the actual architecture string
_META_MAP: List[Tuple[str, str]] = [
    ("{arch}.embedding_length",                 "hidden_size"),
    ("{arch}.block_count",                      "num_hidden_layers"),
    ("{arch}.feed_forward_length",              "intermediate_size"),
    ("{arch}.attention.head_count",             "num_attention_heads"),
    ("{arch}.attention.head_count_kv",          "num_key_value_heads"),
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
            logger.info(job.progress)

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


def _download_tokenizer(hf_repo: str, output_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    for filename in _TOKENIZER_FILES:
        try:
            # token=False → explicit anonymous; avoids 401 from stale env tokens
            cached = hf_hub_download(repo_id=hf_repo, filename=filename, token=False)
            shutil.copy2(cached, output_dir / filename)
            logger.info("Tokenizer file: %s", filename)
        except Exception:
            pass  # File absent from repo — skip silently


# ── Public entry point ───────────────────────────────────────────────────────


def convert_from_gguf(
    job: ConversionJob,
    gguf_path: str,
    hf_tokenizer_repo: str,
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
                           (e.g. "meta-llama/Llama-3.3-70B-Instruct").
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

    # ── 5. Download tokenizer ─────────────────────────────────────────────────
    job.progress = f"Downloading tokenizer from {hf_tokenizer_repo}"
    logger.info("gguf_tokenizer: fetching from %s", hf_tokenizer_repo)
    _download_tokenizer(hf_tokenizer_repo, output)
    logger.info("gguf_tokenizer: done")

    # ── 6. Save weights ───────────────────────────────────────────────────────
    _save_weights(output, remapped, job)

    job.progress = "Done"
    logger.info("gguf_convert: complete → %s", output)
