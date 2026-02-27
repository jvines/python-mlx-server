"""
Generative model manager — wraps mlx_lm for text generation.

Handles:
- Thread-safe lazy loading (one asyncio.Lock per model ID)
- Async streaming via a queue bridge (keeps FastAPI event loop unblocked)
- KV cache controls: max_kv_size, kv_bits, kv_group_size, quantized_kv_start
- Explicit unload + Metal cache clear
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)

# Single-threaded executor: MLX Metal operations serialize on the GPU anyway,
# and this prevents multiple threads from competing for unified memory.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_gen")


def _load_with_progress(model_id: str, path: str) -> Tuple[Any, Any]:
    """
    Load a model via mlx_lm with per-shard progress logging.

    Temporarily wraps mx.load so each .safetensors shard load is intercepted
    and logged with a running byte percentage — same approach as llama.cpp.
    Safe because the generative executor is single-threaded (max_workers=1).
    """
    model_path = Path(path)
    shard_files = sorted(model_path.glob("*.safetensors"))
    total_bytes = sum(f.stat().st_size for f in shard_files)
    total_gb = total_bytes / 1024 ** 3

    logger.info("load: model='%s'  size=%.1f GB  path=%s", model_id, total_gb, path)

    if not shard_files or total_bytes == 0:
        return load(path)

    loaded_bytes = 0
    original_mx_load = mx.load

    def _tracked_load(fpath, *args, **kwargs):
        nonlocal loaded_bytes
        result = original_mx_load(fpath, *args, **kwargs)
        if str(fpath).endswith(".safetensors"):
            loaded_bytes += Path(str(fpath)).stat().st_size
            pct = loaded_bytes / total_bytes * 100
            logger.info(
                "load: '%s'  %5.1f%%  (%.2f / %.2f GB)",
                model_id, pct, loaded_bytes / 1024 ** 3, total_gb,
            )
        return result

    mx.load = _tracked_load
    t0 = time.monotonic()
    try:
        model, tokenizer = load(path)
    finally:
        mx.load = original_mx_load

    logger.info("load: '%s' ready in %.1fs", model_id, time.monotonic() - t0)
    return model, tokenizer


class GenerativeModelManager:
    def __init__(self) -> None:
        self._models: Dict[str, Tuple[Any, Any]] = {}  # id → (model, tokenizer)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_used: Dict[str, float] = {}

    def _lock(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()
        return self._locks[model_id]

    async def load_model(self, model_id: str, path: str) -> Tuple[Any, Any]:
        async with self._lock(model_id):
            if model_id not in self._models:
                loop = asyncio.get_running_loop()
                model, tokenizer = await loop.run_in_executor(
                    _executor, lambda: _load_with_progress(model_id, path)
                )
                self._models[model_id] = (model, tokenizer)
        self._last_used[model_id] = time.monotonic()
        return self._models[model_id]

    def unload(self, model_id: str) -> bool:
        if model_id not in self._models:
            return False
        del self._models[model_id]
        self._last_used.pop(model_id, None)
        mx.metal.clear_cache()
        logger.info(f"Unloaded '{model_id}' and cleared Metal cache")
        return True

    def loaded_models(self) -> List[str]:
        return list(self._models.keys())

    def evict_stale(self, ttl: int) -> List[str]:
        """Unload models idle for longer than ttl seconds. Returns evicted IDs."""
        now = time.monotonic()
        evicted = [
            mid for mid, last in list(self._last_used.items())
            if now - last > ttl
        ]
        for mid in evicted:
            self.unload(mid)
        return evicted

    @staticmethod
    def _build_prompt(tokenizer: Any, messages: List[Dict]) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback for models without a chat template
        return (
            "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            + "\nassistant: "
        )

    async def stream(
        self,
        model_id: str,
        path: str,
        messages: List[Dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.0,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ) -> AsyncGenerator[Any, None]:
        model, tokenizer = await self.load_model(model_id, path)
        prompt = self._build_prompt(tokenizer, messages)

        sampler = make_sampler(temp=temperature, top_p=top_p)
        kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "sampler": sampler}

        if max_kv_size is not None:
            kwargs["max_kv_size"] = max_kv_size
        if kv_bits is not None:
            kwargs["kv_bits"] = kv_bits
            kwargs["kv_group_size"] = kv_group_size
            kwargs["quantized_kv_start"] = quantized_kv_start

        async for chunk in _bridge_to_async(
            lambda: stream_generate(model, tokenizer, prompt, **kwargs)
        ):
            yield chunk


async def _bridge_to_async(make_gen) -> AsyncGenerator[Any, None]:
    """
    Runs a synchronous generator in the thread executor and bridges its
    output to an async generator via a bounded queue.

    The queue size (16) provides backpressure: if the SSE client is slow,
    the worker thread will block on queue.put() rather than building an
    unbounded buffer of tokens in memory.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=16)
    stop_event = threading.Event()
    done_sentinel = object()

    def _worker() -> None:
        try:
            for chunk in make_gen():
                if stop_event.is_set():
                    break
                put_future = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                while True:
                    try:
                        put_future.result(timeout=0.25)
                        break
                    except TimeoutError:
                        if stop_event.is_set():
                            put_future.cancel()
                            return
        except Exception as exc:
            if not stop_event.is_set():
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            try:
                asyncio.run_coroutine_threadsafe(queue.put(done_sentinel), loop).result()
            except Exception:
                # Event loop may already be closed/cancelled.
                pass

    worker_future = loop.run_in_executor(_executor, _worker)

    try:
        while True:
            item = await queue.get()
            if item is done_sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop_event.set()
        if worker_future.done():
            await asyncio.shield(worker_future)


generative_manager = GenerativeModelManager()
