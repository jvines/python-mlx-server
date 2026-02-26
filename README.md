# mlx-server

OpenAI-compatible inference server for Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx).

Designed for running multiple local models side-by-side without multiple llama.cpp processes. Models are registered once by ID, loaded on first use, and evicted after an idle TTL. Full KV cache controls are exposed per-request — critical for Qwen3.5 and other hybrid-attention architectures.

## Features

- **Multi-model registry** — register any local path once, reference by ID forever
- **Lazy loading + TTL eviction** — models stay resident between requests, freed after idle timeout
- **KV cache controls** — `max_kv_size`, `kv_bits`, `kv_group_size` exposed per-request (fixes OOM on Qwen3.5)
- **Three backends** — generative (`mlx-lm`), embeddings (`mlx-embeddings`), vision-language (`mlx-vlm`)
- **Conversion endpoint** — convert HuggingFace repos or local GGUFs to MLX format via async jobs
- **OpenAI-compatible** — `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`
- **Streaming SSE** — `"stream": true` returns Server-Sent Events

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/jvines/python-mlx-server.git && cd python-mlx-server
uv sync
```

## Quick start

```bash
# Start the server (port 1234)
uv run mlx-server

# Register a model you already have locally
curl -X POST http://localhost:1234/v1/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "id": "qwen3-8b",
    "path": "/path/to/qwen3-8b-mlx",
    "type": "generative"
  }'

# Chat
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Explain KV cache in one sentence."}],
    "max_tokens": 128,
    "stream": false
  }'
```

## Configuration

All settings use the `MLX_` environment variable prefix:

| Variable | Default | Description |
|---|---|---|
| `MLX_HOST` | `127.0.0.1` | Bind address |
| `MLX_PORT` | `1234` | Port |
| `MLX_LOG_LEVEL` | `INFO` | Log level |
| `MLX_MODEL_REGISTRY_PATH` | `~/.config/mlx-server/registry.json` | Persisted registry |
| `MLX_MODEL_DIR` | `~/.cache/mlx-server/models` | Default conversion output root |
| `MLX_MODEL_TTL_SECONDS` | `1800` | Seconds before idle models are evicted |

```bash
MLX_PORT=9000 MLX_MODEL_TTL_SECONDS=3600 uv run mlx-server

# Expose on your LAN (only if you need remote access)
MLX_HOST=0.0.0.0 uv run mlx-server
```

## API reference

### Models

```bash
# Register a model
POST /v1/models/register
{
  "id": "my-model",
  "path": "/absolute/path/to/model",
  "type": "generative" | "embedding" | "vlm",
  "mmproj": "/path/to/mmproj.safetensors"  # VLM only, optional
}

# List registered models
GET /v1/models

# Get one model
GET /v1/models/{id}

# Eagerly load (pre-warm) a model
POST /v1/models/{id}/load

# Evict from memory (stays registered)
DELETE /v1/models/{id}

# Remove from registry entirely
DELETE /v1/models/{id}/unregister
```

### Chat completions

```bash
POST /v1/chat/completions
{
  "model": "my-model",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": null,
  "stream": false,

  # KV cache controls (generative models only)
  "max_kv_size": 8192,      # cap KV cache to N tokens (circular buffer)
  "kv_bits": 4,             # quantize KV cache (4 or 8) — 2-4x memory reduction
  "kv_group_size": 64,
  "quantized_kv_start": 0
}
```

**Streaming example:**
```bash
curl -N -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

**Fixing Qwen3.5 KV OOM:**
```json
{
  "model": "qwen3-35b",
  "messages": [...],
  "max_kv_size": 4096,
  "kv_bits": 4
}
```

### Embeddings

```bash
POST /v1/embeddings
{
  "model": "bge-m3",
  "input": "text to embed"          # string or list of strings
}
```

### Conversion

Conversions run as background jobs. Poll or stream progress.

**Convert from HuggingFace:**
```bash
POST /v1/convert/hf
{
  "hf_repo": "Qwen/Qwen3-8B",
  "output_path": "/path/to/output",
  "model_type": "generative",
  "quantize": true,
  "q_bits": 4,
  "register_as": "qwen3-8b"   # optional: auto-register on completion
}
```

**Convert a local GGUF:**
```bash
POST /v1/convert/gguf
{
  "gguf_path": "/path/to/model.gguf",
  "hf_tokenizer_repo": "meta-llama/Llama-3.3-70B-Instruct",
  "output_path": "/path/to/output",
  "quantize": false,           # false for Q8_0 (already MLX-native); true for K-quants
  "register_as": "llama-70b"
}
```

Supported GGUF architectures: `llama`, `qwen2`, `qwen2moe`, `mistral`, `mistral3`, `gemma`, `gemma2`

> **Note on K-quant GGUFs (Q4_K_M, Q5_K_S, etc.):** `mx.load` dequantizes these to float16.
> A 70B K-quant model requires ~140 GB intermediate RAM — use the HF path for 70B models.
> Q8_0 GGUFs stay natively quantized throughout.

**Poll job status:**
```bash
GET /v1/convert/{job_id}
GET /v1/convert              # list all jobs
GET /v1/convert/{job_id}/stream   # SSE live progress
```

## Multimodal (VLM)

Register with `"type": "vlm"` and include image URLs in the message content:

```json
{
  "model": "my-vlm",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "https://..."}}
    ]
  }]
}
```

## Development

```bash
uv sync
uv run pytest tests/ -v

# Smoke test against a live server
uv run mlx-server &
uv run python smoke_test.py --test inference
uv run python smoke_test.py --test hf
uv run python smoke_test.py --test gguf
```

## Project structure

```
src/mlx_server/
├── config.py          # Settings (MLX_ env prefix)
├── registry.py        # Model registry — persisted JSON, ID → path + type
├── managers/
│   ├── generative.py  # mlx-lm backend (text LLMs)
│   ├── embedding.py   # mlx-embeddings backend
│   └── vlm.py         # mlx-vlm backend (vision-language)
├── conversion/
│   ├── jobs.py        # Async job tracking
│   ├── hf.py          # HuggingFace → MLX conversion
│   └── gguf.py        # GGUF → MLX conversion (weight remapping)
└── api/
    ├── app.py         # FastAPI app
    ├── chat.py        # POST /v1/chat/completions
    ├── embeddings.py  # POST /v1/embeddings
    ├── models.py      # /v1/models/*
    └── convert.py     # /v1/convert/*
tests/
├── test_api.py        # API + manager tests (mocked, no models needed)
└── test_conversion.py # Conversion job + weight remapping tests
smoke_test.py          # End-to-end tests against a live server
```
