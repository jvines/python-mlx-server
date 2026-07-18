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
import inspect
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
        # Count of in-flight generations per model, so eviction/unload never
        # frees a model out from under an active stream. Mutated only on the
        # event loop, so no extra locking is required.
        self._in_use: Dict[str, int] = {}

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

    def unload(self, model_id: str, force: bool = False) -> bool:
        if model_id not in self._models:
            return False
        if not force and self._in_use.get(model_id, 0) > 0:
            logger.info(
                "Not unloading '%s': %d generation(s) still in flight",
                model_id, self._in_use[model_id],
            )
            return False
        del self._models[model_id]
        self._last_used.pop(model_id, None)
        self._locks.pop(model_id, None)
        mx.metal.clear_cache()
        logger.info(f"Unloaded '{model_id}' and cleared Metal cache")
        return True

    def loaded_models(self) -> List[str]:
        return list(self._models.keys())

    def evict_stale(self, ttl: int) -> List[str]:
        """Unload models idle for longer than ttl seconds. Returns evicted IDs.

        Models with an in-flight generation are never evicted, even if their
        last-used timestamp looks stale.
        """
        now = time.monotonic()
        evicted = []
        for mid, last in list(self._last_used.items()):
            if now - last > ttl and self._in_use.get(mid, 0) == 0:
                if self.unload(mid):
                    evicted.append(mid)
        return evicted

    @staticmethod
    def _build_prompt(
        tokenizer: Any,
        messages: List[Dict],
        enable_thinking: Optional[bool] = None,
    ) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            template_kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking is not None:
                supports_enable_thinking = False
                try:
                    sig = inspect.signature(tokenizer.apply_chat_template)
                    params = sig.parameters.values()
                    supports_enable_thinking = (
                        "enable_thinking" in sig.parameters
                        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
                    )
                except (TypeError, ValueError):
                    # Some tokenizer wrappers may not expose a Python signature.
                    # Try passing the arg and fall back if unsupported.
                    supports_enable_thinking = True

                if supports_enable_thinking:
                    template_kwargs["enable_thinking"] = enable_thinking
                else:
                    logger.info(
                        "Tokenizer does not support enable_thinking; using default behavior"
                    )

            try:
                return tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # The introspection above can be fooled by wrapped tokenizers.
                # If the template actually rejects enable_thinking, retry
                # without it rather than 500 the request.
                if "enable_thinking" in template_kwargs:
                    logger.info(
                        "apply_chat_template rejected enable_thinking; "
                        "retrying with default thinking behavior"
                    )
                    template_kwargs.pop("enable_thinking")
                    return tokenizer.apply_chat_template(messages, **template_kwargs)
                raise
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
        enable_thinking: Optional[bool] = None,
    ) -> AsyncGenerator[Any, None]:
        model, tokenizer = await self.load_model(model_id, path)
        prompt = self._build_prompt(tokenizer, messages, enable_thinking=enable_thinking)

        sampler = make_sampler(temp=temperature, top_p=top_p)
        kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "sampler": sampler}

        if max_kv_size is not None:
            kwargs["max_kv_size"] = max_kv_size
        if kv_bits is not None:
            kwargs["kv_bits"] = kv_bits
            kwargs["kv_group_size"] = kv_group_size
            kwargs["quantized_kv_start"] = quantized_kv_start

        self._in_use[model_id] = self._in_use.get(model_id, 0) + 1
        try:
            async for chunk in _bridge_to_async(
                lambda: stream_generate(model, tokenizer, prompt, **kwargs)
            ):
                yield chunk
        finally:
            self._in_use[model_id] = max(0, self._in_use.get(model_id, 1) - 1)
            # Refresh idle timer at completion so a long generation isn't judged
            # stale from the timestamp taken when it started.
            self._last_used[model_id] = time.monotonic()


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

    def _put_stop_aware(item: Any) -> bool:
        """Enqueue ``item`` from the worker thread without ever blocking
        forever. Returns True if enqueued, False if we bailed because the
        consumer asked us to stop (or the loop is gone). Every cross-thread put
        goes through here — including the terminal sentinel — so a full queue
        and a departed consumer can never wedge the single worker thread."""
        try:
            fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        except RuntimeError:
            return False  # loop already closed
        while True:
            try:
                fut.result(timeout=0.25)
                return True
            except TimeoutError:
                if stop_event.is_set():
                    fut.cancel()
                    return False
            except Exception:
                return False  # loop closing/closed

    def _worker() -> None:
        try:
            for chunk in make_gen():
                if stop_event.is_set():
                    break
                if not _put_stop_aware(chunk):
                    return
        except Exception as exc:
            if not stop_event.is_set():
                _put_stop_aware(exc)
        finally:
            # Stop-aware, so if the consumer has already left this returns
            # promptly instead of blocking on a full, abandoned queue.
            _put_stop_aware(done_sentinel)

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
        # Drain anything buffered so a worker blocked on queue.put() gets a free
        # slot, completes its put, observes stop_event, and exits — instead of
        # wedging the sole executor thread for every future request.
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Do NOT hard-await the worker here. On the single-thread executor it may
        # be queued behind another in-flight generation, and blocking this
        # (possibly cancelled) teardown on it would stall until that one
        # finishes — and an asyncio.shield would make it uncancellable at
        # shutdown. The drain + stop_event already guarantee the worker
        # terminates on its own; we only consume any exception it surfaced so it
        # isn't reported as "never retrieved".
        def _consume(fut) -> None:
            if not fut.cancelled():
                fut.exception()

        if worker_future.done():
            _consume(worker_future)
        else:
            worker_future.add_done_callback(_consume)


generative_manager = GenerativeModelManager()
