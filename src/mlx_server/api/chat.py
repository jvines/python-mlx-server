"""
POST /v1/chat/completions

Supports:
- Streaming (stream=true) via Server-Sent Events
- Non-streaming (stream=false) blocking response
- Multimodal content (text + image_url) for VLM models
- KV cache controls: max_kv_size, kv_bits, kv_group_size, quantized_kv_start
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from pathlib import Path

from ..managers import generative_manager, vlm_manager
from ..registry import registry, ModelEntry

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


MessageContent = Union[str, List[Union[TextContent, ImageContent]]]


class Message(BaseModel):
    role: str
    content: MessageContent

    def as_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            p.text for p in self.content if isinstance(p, TextContent)
        )

    def image_urls(self) -> List[str]:
        if isinstance(self.content, str):
            return []
        return [
            p.image_url.url for p in self.content if isinstance(p, ImageContent)
        ]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]

    # Sampling
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=512, ge=1)
    stream: Optional[bool] = False

    # KV cache controls — exposed to fix OOM on long-context / new architectures
    # max_kv_size: cap the KV cache to N tokens (circular buffer, old tokens evicted)
    max_kv_size: Optional[int] = Field(default=None, ge=64)
    # kv_bits: quantise the KV cache (4 or 8). Reduces KV memory 2-4x.
    kv_bits: Optional[Literal[4, 8]] = None
    kv_group_size: int = Field(default=64, ge=1)
    quantized_kv_start: int = Field(default=0, ge=0)

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages must not be empty")
        return v


class ChatChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tps: Optional[float] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: Usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    model: str,
    created: int,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _collect_images(messages: List[Message]) -> List[str]:
    images: List[str] = []
    for msg in messages:
        images.extend(msg.image_urls())
    return images


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    entry = registry.get(request.model)
    if entry is None:
        # Accept a bare file-system path as the model field — no registration needed
        p = Path(request.model)
        if p.exists():
            if not p.is_absolute():
                raise HTTPException(
                    status_code=400,
                    detail="When passing a model path directly, it must be absolute.",
                )
            entry = ModelEntry.model_construct(
                path=str(p.resolve()),
                type="generative",
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
    if entry.type == "embedding":
        raise HTTPException(
            status_code=400,
            detail=f"'{request.model}' is an embedding model. Use POST /v1/embeddings.",
        )

    messages_dicts = [
        {"role": m.role, "content": m.as_text()} for m in request.messages
    ]

    kv_kwargs = dict(
        max_tokens=request.max_tokens,
        temperature=0.7 if request.temperature is None else request.temperature,
        top_p=0.0 if request.top_p is None else request.top_p,
        max_kv_size=request.max_kv_size,
        kv_bits=request.kv_bits,
        kv_group_size=request.kv_group_size,
        quantized_kv_start=request.quantized_kv_start,
    )

    if entry.type == "vlm":
        images = _collect_images(request.messages)
        gen = vlm_manager.stream(
            request.model, entry.path, messages_dicts, images=images or None, **kv_kwargs
        )
    else:
        gen = generative_manager.stream(
            request.model, entry.path, messages_dicts, **kv_kwargs
        )

    if request.stream:
        return StreamingResponse(
            _sse_stream(gen, request.model),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    return await _blocking_response(gen, request.model)


async def _sse_stream(gen, model: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Opening chunk — signals role to the client
    yield _make_chunk(chunk_id, model, created, {"role": "assistant", "content": ""})

    last_response = None
    async for response in gen:
        last_response = response
        if response.text:
            yield _make_chunk(chunk_id, model, created, {"content": response.text})

    # Closing chunk with finish_reason and optional usage
    usage_data: Dict[str, Any] = {}
    finish_reason = "stop"
    if last_response is not None:
        finish_reason = last_response.finish_reason or "stop"
        usage_data = {
            "prompt_tokens": last_response.prompt_tokens,
            "completion_tokens": last_response.generation_tokens,
            "total_tokens": last_response.prompt_tokens + last_response.generation_tokens,
            "completion_tps": round(last_response.generation_tps, 2),
        }

    closing = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": usage_data,
    }
    yield f"data: {json.dumps(closing)}\n\n"
    yield "data: [DONE]\n\n"


async def _blocking_response(gen, model: str) -> ChatCompletionResponse:
    full_text = ""
    last_response = None
    async for response in gen:
        full_text += response.text
        last_response = response

    prompt_tokens = last_response.prompt_tokens if last_response else 0
    completion_tokens = last_response.generation_tokens if last_response else 0
    finish_reason = (last_response.finish_reason or "stop") if last_response else "stop"
    completion_tps = round(last_response.generation_tps, 2) if last_response else None

    return ChatCompletionResponse(
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message={"role": "assistant", "content": full_text},
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            completion_tps=completion_tps,
        ),
    )
