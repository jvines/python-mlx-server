"""Shared request-resolution helpers for the API routers."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import HTTPException

from ..registry import ModelEntry, registry


def resolve_model_entry(model_id: str, default_type: str) -> ModelEntry:
    """Resolve a model reference to a :class:`ModelEntry`.

    Accepts either a registered model ID or a bare absolute filesystem path to
    an MLX model directory. A bare path needs no prior registration and is
    treated as ``default_type`` (so ``/v1/chat/completions`` assumes generative
    and ``/v1/embeddings`` assumes embedding). Raises ``HTTPException`` with an
    OpenAI-style status when the reference can't be resolved.
    """
    entry = registry.get(model_id)
    if entry is not None:
        return entry

    p = Path(model_id)
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model_id}' is not registered and the path does not exist. "
                "Register it with POST /v1/models/register or pass an absolute path."
            ),
        )
    if not p.is_absolute():
        raise HTTPException(
            status_code=400,
            detail="When passing a model path directly, it must be absolute.",
        )
    if not p.is_dir() or not (p / "config.json").exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Path '{model_id}' is not an MLX model directory (no config.json). "
                "Pass a converted model directory, or register the model explicitly "
                "with its type via POST /v1/models/register."
            ),
        )
    return ModelEntry.model_construct(
        path=str(p.resolve()),
        type=default_type,
        created=int(time.time()),
    )
