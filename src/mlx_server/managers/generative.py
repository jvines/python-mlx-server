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
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)

# Single-threaded executor: MLX Metal operations serialize on the GPU anyway,
# and this prevents multiple threads from competing for unified memory.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_gen")


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
                logger.info(f"Loading generative model '{model_id}' from {path}")
                loop = asyncio.get_running_loop()
                model, tokenizer = await loop.run_in_executor(
                    _executor, lambda: load(path)
                )
                self._models[model_id] = (model, tokenizer)
                logger.info(f"Model '{model_id}' ready")
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

    def _worker() -> None:
        try:
            for chunk in make_gen():
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    worker_future = loop.run_in_executor(_executor, _worker)

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        await asyncio.shield(worker_future)


generative_manager = GenerativeModelManager()
