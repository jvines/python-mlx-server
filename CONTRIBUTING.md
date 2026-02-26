# Contributing

## Dev setup

```bash
uv sync           # installs all deps including dev group
uv run pytest     # run tests (no models or server needed)
```

Tests mock all model loading — you can run them without any MLX models installed.

## Running the server locally

```bash
uv run mlx-server
# Interactive API docs at http://localhost:8080/docs
```

## Testing against real models

```bash
uv run mlx-server &
uv run python smoke_test.py --test inference   # downloads SmolLM2-135M (~270 MB)
uv run python smoke_test.py --test hf          # converts SmolLM2-135M to 4-bit
uv run python smoke_test.py --test gguf        # converts a local GGUF (edit path in smoke_test.py)
```

## Code structure

```
src/mlx_server/
├── config.py          # All settings — add new options here
├── registry.py        # ModelRegistry — persists to JSON, validates paths
├── managers/          # One file per backend
│   ├── generative.py  # mlx-lm
│   ├── embedding.py   # mlx-embeddings
│   └── vlm.py         # mlx-vlm
├── conversion/        # Background conversion jobs
│   ├── jobs.py        # JobManager + ConversionJob dataclass
│   ├── hf.py          # HF repo → MLX
│   └── gguf.py        # GGUF → MLX (weight remapping, config reconstruction)
└── api/
    ├── app.py         # FastAPI app — mount new routers here
    ├── chat.py        # /v1/chat/completions
    ├── embeddings.py  # /v1/embeddings
    ├── models.py      # /v1/models/*
    └── convert.py     # /v1/convert/*
```

## Adding a new model backend

1. Create `src/mlx_server/managers/mybackend.py` — follow the pattern in `generative.py`:
   - `load_model(model_id, path)` — async, returns cached model
   - `stream(model_id, path, ...)` — async generator yielding response chunks
   - Use `_bridge_to_async` from `generative.py` to run sync inference off the event loop

2. Export from `src/mlx_server/managers/__init__.py`

3. Wire into the relevant API router (`chat.py`, `embeddings.py`, etc.)

## Adding a new GGUF architecture

GGUF weight remapping lives in `src/mlx_server/conversion/gguf.py`.

1. Add the architecture name to `_ARCH_MODEL_TYPE` and `_ARCH_CLASS`:
   ```python
   _ARCH_MODEL_TYPE["myarch"] = "my_hf_model_type"
   _ARCH_CLASS["my_hf_model_type"] = "MyModelForCausalLM"
   ```

2. If the architecture uses non-standard layer key names, add them to `_LAYER_MAP`.

3. Add any architecture-specific config fields in `_build_config()`.

4. Add a unit test in `tests/test_conversion.py` using `_remap_weights` and `_build_config` directly.

## Key design decisions

**Async generator bridge** (`managers/generative.py`): `mlx_lm.stream_generate` is synchronous. It runs in a `ThreadPoolExecutor` and feeds tokens into an `asyncio.Queue(maxsize=16)`. The `maxsize` provides backpressure — inference pauses if the client is reading slowly. The async generator on the FastAPI side consumes from the queue.

**`stream()` is an async generator, not a coroutine.** Never `await manager.stream(...)` — iterate it with `async for`.

**mlx-lm 0.30.7+ breaking change**: `generate()` and `stream_generate()` no longer accept `temp=`. Use `sampler=make_sampler(temp=t, top_p=p)` from `mlx_lm.sample_utils`.

**Registry path validation**: `ModelEntry` validates that `path` exists at registration time. In tests, use `ModelEntry.model_construct(...)` to bypass validation.

**Single `ThreadPoolExecutor(max_workers=1)` per manager**: Metal serializes GPU work anyway. Multiple workers would cause memory contention without throughput benefit.

## Making changes

- Keep `api/` thin — request parsing and response shaping only, no inference logic
- Keep `managers/` backend-specific — no FastAPI imports
- All new settings go in `config.py` with an `MLX_` env prefix
- New endpoints get their own router file and are mounted in `api/app.py`
