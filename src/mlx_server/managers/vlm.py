"""
VLM manager — wraps mlx_vlm for vision-language model inference.

Supports text-only and multimodal (text + image) requests.
Images may be local filesystem paths or HTTP/HTTPS URLs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_vlm")


def _check_mlx_vlm() -> None:
    try:
        import mlx_vlm  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "mlx-vlm is not installed. "
            "Run: uv add mlx-vlm"
        )


class VLMModelManager:
    def __init__(self) -> None:
        self._models: Dict[str, Tuple[Any, Any]] = {}  # id → (model, processor)
        self._configs: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_used: Dict[str, float] = {}

    def _lock(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()
        return self._locks[model_id]

    async def load_model(self, model_id: str, path: str) -> Tuple[Any, Any]:
        async with self._lock(model_id):
            if model_id not in self._models:
                _check_mlx_vlm()
                from mlx_vlm import load as vlm_load
                from mlx_vlm.utils import load_config

                logger.info(f"Loading VLM '{model_id}' from {path}")
                loop = asyncio.get_running_loop()

                def _load():
                    model, processor = vlm_load(path)
                    config = load_config(path)
                    return model, processor, config

                model, processor, config = await loop.run_in_executor(_executor, _load)
                self._models[model_id] = (model, processor)
                self._configs[model_id] = config
                logger.info(f"VLM '{model_id}' ready")
        self._last_used[model_id] = time.monotonic()
        return self._models[model_id]

    def unload(self, model_id: str) -> bool:
        if model_id not in self._models:
            return False
        del self._models[model_id]
        self._configs.pop(model_id, None)
        self._last_used.pop(model_id, None)
        mx.metal.clear_cache()
        logger.info(f"Unloaded VLM '{model_id}'")
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
    def _build_prompt(
        processor: Any,
        config: Any,
        messages: List[Dict],
        num_images: int,
    ) -> str:
        from mlx_vlm.prompt_utils import apply_chat_template

        # Use the last user message as the prompt; prepend earlier turns as context.
        # Full multi-turn history for VLMs is model-specific — this covers the
        # common single/dual-turn case cleanly.
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            raise ValueError("At least one user message is required.")

        last_user = user_messages[-1]
        text = (
            last_user["content"]
            if isinstance(last_user["content"], str)
            else " ".join(
                p["text"] for p in last_user["content"] if p.get("type") == "text"
            )
        )
        return apply_chat_template(processor, config, text, num_images=num_images)

    async def stream(
        self,
        model_id: str,
        path: str,
        messages: List[Dict],
        images: Optional[List[str]] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ) -> AsyncGenerator[Any, None]:
        _check_mlx_vlm()
        from mlx_vlm import stream_generate as vlm_stream

        model, processor = await self.load_model(model_id, path)
        config = self._configs[model_id]

        num_images = len(images) if images else 0
        prompt = self._build_prompt(processor, config, messages, num_images)

        kwargs: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if images:
            kwargs["image"] = images if len(images) > 1 else images[0]
        if max_kv_size is not None:
            kwargs["max_kv_size"] = max_kv_size
        if kv_bits is not None:
            kwargs["kv_bits"] = kv_bits
            kwargs["kv_group_size"] = kv_group_size
            kwargs["quantized_kv_start"] = quantized_kv_start

        from .generative import _bridge_to_async

        async for chunk in _bridge_to_async(
            lambda: vlm_stream(model, processor, prompt, **kwargs)
        ):
            yield chunk


vlm_manager = VLMModelManager()
