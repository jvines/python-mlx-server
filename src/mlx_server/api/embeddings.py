"""
POST /v1/embeddings

OpenAI-compatible embeddings endpoint backed by mlx_embeddings.
"""

from __future__ import annotations

import base64
import logging
from typing import List, Literal, Optional, Union

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from ..managers import embedding_manager
from .deps import resolve_model_entry

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

    @field_validator("input")
    @classmethod
    def _input_within_limits(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        items = [v] if isinstance(v, str) else v
        if len(items) > 2048:
            raise ValueError("input has too many items (max 2048)")
        if sum(len(s) for s in items) > 2_000_000:
            raise ValueError("input is too large (max 2,000,000 characters total)")
        return v


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    # A vector of floats, or its base64-encoded little-endian float32 form when
    # encoding_format="base64" (matches the OpenAI response contract).
    embedding: Union[List[float], str]
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
# Helpers
# ---------------------------------------------------------------------------


def _encode_base64(vec: List[float]) -> str:
    """OpenAI-compatible base64 encoding: little-endian float32 bytes, base64."""
    arr = np.asarray(vec, dtype="<f4")
    return base64.b64encode(arr.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    input_count = 1 if isinstance(request.input, str) else len(request.input)
    logger.info(
        "embeddings_request model=%s inputs=%d encoding_format=%s",
        request.model, input_count, request.encoding_format,
    )

    if request.dimensions is not None:
        logger.info(
            "dimensions=%s requested but not enforced (model-determined)",
            request.dimensions,
        )

    entry = resolve_model_entry(request.model, "embedding")
    if entry.type != "embedding":
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{request.model}' is a {entry.type} model. "
                "Use POST /v1/chat/completions for generative/vlm models."
            ),
        )

    texts = request.input
    if isinstance(texts, list) and not texts:
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
    if isinstance(texts, str):
        total_tokens = len(texts.split())
    else:
        total_tokens = sum(len(t.split()) for t in texts)

    if request.encoding_format == "base64":
        data = [
            EmbeddingObject(embedding=_encode_base64(vec), index=i)
            for i, vec in enumerate(vectors)
        ]
    else:
        data = [
            EmbeddingObject(embedding=vec, index=i)
            for i, vec in enumerate(vectors)
        ]

    return EmbeddingResponse(
        model=request.model,
        data=data,
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )
