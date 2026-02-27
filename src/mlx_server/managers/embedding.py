"""
Embedding model manager — wraps mlx_embeddings.

Returns L2-normalised float arrays compatible with the OpenAI
/v1/embeddings response format.
"""

from __future__ import annotations

import asyncio
import logging
import numbers
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx

logger = logging.getLogger(__name__)

# Separate executor for embeddings so loading/inference doesn't block the
# generative executor (and vice-versa).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_emb")


def _check_mlx_embeddings() -> None:
    try:
        import mlx_embeddings  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "mlx-embeddings is not installed. "
            "Run: uv add mlx-embeddings"
        )


def _normalize_embedding_output(result: Any) -> List[List[float]]:
    """
    Normalize mlx_embeddings outputs to list[list[float]].

    Newer mlx_embeddings returns structured dataclasses (e.g. BaseModelOutput)
    with fields like text_embeds/pooler_output instead of a raw mx.array.
    """
    source = result
    kind = "raw"

    if isinstance(result, dict):
        for key in ("text_embeds", "pooler_output", "embeddings", "last_hidden_state"):
            if result.get(key) is not None:
                source = result[key]
                kind = key
                break
    else:
        for attr in ("text_embeds", "pooler_output", "embeddings", "last_hidden_state"):
            if hasattr(result, attr):
                value = getattr(result, attr)
                if value is not None:
                    source = value
                    kind = attr
                    break
        if kind == "raw" and isinstance(result, (list, tuple)) and result:
            source = result[0]
            kind = "sequence[0]"

    # last_hidden_state is typically (batch, seq, hidden); mean-pool as fallback
    if kind == "last_hidden_state" and hasattr(source, "ndim") and source.ndim == 3:
        source = mx.mean(source, axis=1)

    values = source.tolist() if hasattr(source, "tolist") else source
    if not isinstance(values, list):
        raise RuntimeError(f"Unexpected embedding output type: {type(values)}")

    if not values:
        return []

    # Single embedding vector (1-D) -> wrap to batch shape.
    if isinstance(values[0], numbers.Number):
        return [values]

    return values


class EmbeddingModelManager:
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
                _check_mlx_embeddings()
                from mlx_embeddings import load as emb_load

                logger.info(f"Loading embedding model '{model_id}' from {path}")
                loop = asyncio.get_running_loop()
                model, tokenizer = await loop.run_in_executor(
                    _executor, lambda: emb_load(path)
                )
                self._models[model_id] = (model, tokenizer)
                logger.info(f"Embedding model '{model_id}' ready")
        self._last_used[model_id] = time.monotonic()
        return self._models[model_id]

    def unload(self, model_id: str) -> bool:
        if model_id not in self._models:
            return False
        del self._models[model_id]
        self._last_used.pop(model_id, None)
        mx.metal.clear_cache()
        logger.info(f"Unloaded embedding model '{model_id}'")
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

    async def embed(
        self,
        model_id: str,
        path: str,
        texts: Union[str, List[str]],
    ) -> List[List[float]]:
        """
        Returns a list of embedding vectors (one per input text).
        Vectors are L2-normalised float32 lists, ready for the API response.
        """
        _check_mlx_embeddings()
        from mlx_embeddings import generate as emb_generate

        model, tokenizer = await self.load_model(model_id, path)

        def _generate() -> List[List[float]]:
            # Some tokenizers (e.g. Qwen2Tokenizer in recent transformers)
            # don't expose batch_encode_plus; fallback to per-item embedding.
            if isinstance(texts, list) and not hasattr(tokenizer, "batch_encode_plus"):
                logger.info(
                    "Tokenizer for '%s' has no batch_encode_plus; embedding batch item-by-item",
                    model_id,
                )
                vectors: List[List[float]] = []
                for text in texts:
                    item = _normalize_embedding_output(
                        emb_generate(model, tokenizer, text)
                    )
                    if not item:
                        continue
                    vectors.append(item[0])
                return vectors
            return _normalize_embedding_output(emb_generate(model, tokenizer, texts))

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, _generate)


embedding_manager = EmbeddingModelManager()
