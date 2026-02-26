"""
POST /v1/embeddings

OpenAI-compatible embeddings endpoint backed by mlx_embeddings.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pathlib import Path

from ..managers import embedding_manager
from ..registry import registry, ModelEntry

router = APIRouter()


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
    entry = registry.get(request.model)
    if entry is None:
        p = Path(request.model)
        if p.exists():
            entry = ModelEntry.model_construct(path=str(p), type="embedding")
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
        raise HTTPException(status_code=500, detail=str(exc))

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
