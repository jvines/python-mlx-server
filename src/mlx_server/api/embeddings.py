"""
POST /v1/embeddings

OpenAI-compatible embeddings endpoint backed by mlx_embeddings.
"""

from __future__ import annotations

import logging
import time
from typing import List, Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pathlib import Path

from ..managers import embedding_manager
from ..registry import registry, ModelEntry

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None  # accepted but not enforced (model-determined)


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: EmbeddingUsage


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    if request.encoding_format != "float":
        raise HTTPException(
            status_code=422,
            detail="Only encoding_format='float' is currently supported.",
        )
    if request.dimensions is not None:
        raise HTTPException(
            status_code=422,
            detail="dimensions is not currently supported by mlx_embeddings.",
        )

    entry = registry.get(request.model)
    if entry is None:
        p = Path(request.model)
        if p.exists():
            if not p.is_absolute():
                raise HTTPException(
                    status_code=400,
                    detail="When passing a model path directly, it must be absolute.",
                )
            entry = ModelEntry.model_construct(
                path=str(p.resolve()),
                type="embedding",
                created=int(time.time()),
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model '{request.model}' is not registered and the path does not exist. "
                    "Register it with POST /v1/models/register or pass an absolute path."
                ),
            )
    if entry.type != "embedding":
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{request.model}' is a {entry.type} model. "
                "Use POST /v1/chat/completions for generative/vlm models."
            ),
        )

    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        raise HTTPException(status_code=422, detail="input must not be empty")

    try:
        vectors = await embedding_manager.embed(request.model, entry.path, texts)
    except Exception as exc:
        logger.exception("Embedding generation failed for model '%s'", request.model)
        raise HTTPException(
            status_code=500,
            detail="Embedding generation failed. Check server logs for details.",
        ) from exc

    # Rough token estimate — embeddings libraries don't always expose exact counts
    total_tokens = sum(len(t.split()) for t in texts)

    return EmbeddingResponse(
        model=request.model,
        data=[
            EmbeddingObject(embedding=vec, index=i)
            for i, vec in enumerate(vectors)
        ],
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )
