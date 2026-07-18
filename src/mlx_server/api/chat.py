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
import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..managers import generative_manager, vlm_manager
from .deps import resolve_model_entry

router = APIRouter()
logger = logging.getLogger(__name__)


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
    max_tokens: Optional[int] = Field(default=512, ge=1, le=131072)
    stream: Optional[bool] = False
    # Number of completions. Only n=1 is supported; other values are rejected
    # rather than silently ignored.
    n: Optional[int] = Field(default=1, ge=1)
    # Qwen3.5-compatible explicit thinking control.
    # If None, tokenizer/model default behavior is used.
    enable_thinking: Optional[bool] = None

    # KV cache controls — exposed to fix OOM on long-context / new architectures
    # max_kv_size: cap the KV cache to N tokens (circular buffer, old tokens evicted)
    max_kv_size: Optional[int] = Field(default=None, ge=64, le=1_048_576)
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
    logger.info(
        "chat_request model=%s stream=%s messages=%d max_tokens=%s enable_thinking=%s",
        request.model,
        bool(request.stream),
        len(request.messages),
        request.max_tokens,
        request.enable_thinking,
    )
    if request.n is not None and request.n != 1:
        raise HTTPException(
            status_code=400,
            detail="Only n=1 is supported.",
        )

    entry = resolve_model_entry(request.model, "generative")
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
        if request.enable_thinking is not None:
            logger.info("enable_thinking is not supported for VLM models; ignoring")
        images = _collect_images(request.messages)
        # VLMModelManager.stream has no top_p parameter (mlx_vlm samples on
        # temperature only). Drop it here instead of splatting it in and
        # raising TypeError on every VLM request.
        vlm_kwargs = {k: v for k, v in kv_kwargs.items() if k != "top_p"}
        gen = vlm_manager.stream(
            request.model, entry.path, messages_dicts, images=images or None, **vlm_kwargs
        )
    else:
        gen = generative_manager.stream(
            request.model,
            entry.path,
            messages_dicts,
            enable_thinking=request.enable_thinking,
            **kv_kwargs,
        )

    if request.stream:
        return await _streaming_response(gen, request.model)

    try:
        return await _blocking_response(gen, request.model)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Generation failed for model '%s'", request.model)
        raise HTTPException(
            status_code=500,
            detail="Generation failed. Check server logs for details.",
        ) from exc


async def _streaming_response(gen, model: str) -> StreamingResponse:
    """Prime the generator so load/setup errors surface as a normal HTTP error
    (before streaming headers are committed), then stream the rest as SSE."""
    agen = gen.__aiter__()
    try:
        first = await agen.__anext__()
    except StopAsyncIteration:
        first = None
    except Exception as exc:
        logger.exception("Generation failed before streaming for model '%s'", model)
        raise HTTPException(
            status_code=503,
            detail="Generation failed to start. Check server logs for details.",
        ) from exc
    return StreamingResponse(
        _sse_stream(agen, first, model),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


async def _sse_stream(agen, first, model: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Opening chunk — signals role to the client
    yield _make_chunk(chunk_id, model, created, {"role": "assistant", "content": ""})

    last_response = first
    try:
        try:
            if first is not None and first.text:
                yield _make_chunk(chunk_id, model, created, {"content": first.text})
            async for response in agen:
                last_response = response
                if response.text:
                    yield _make_chunk(chunk_id, model, created, {"content": response.text})
        except Exception:
            # Failure after headers were already committed: the status code is
            # fixed at 200, so emit an OpenAI-style error event and terminate the
            # stream cleanly instead of leaving the client on a truncated body.
            logger.exception("Streaming generation failed mid-stream for model '%s'", model)
            err = {
                "error": {
                    "message": "Generation failed mid-stream. Check server logs for details.",
                    "type": "server_error",
                }
            }
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Closing chunk with finish_reason and optional usage
        usage_data: Dict[str, Any] = {}
        finish_reason = "stop"
        if last_response is not None:
            # mlx_vlm's GenerationResult has no finish_reason attribute (mlx_lm's
            # GenerationResponse does); read it defensively so both backends work.
            finish_reason = getattr(last_response, "finish_reason", None) or "stop"
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
    finally:
        # Deterministically tear down the generator chain (bridge worker +
        # in-use counter) even on client disconnect: async-for does not aclose
        # its iterator on GeneratorExit, so without this the bridge worker could
        # be left running and the in-use count leaked until GC.
        await agen.aclose()


async def _blocking_response(gen, model: str) -> ChatCompletionResponse:
    full_text = ""
    last_response = None
    agen = gen.__aiter__()
    try:
        async for response in agen:
            full_text += response.text
            last_response = response
    finally:
        # Ensure the bridge worker + in-use counter are torn down even if the
        # request is cancelled mid-generation (client disconnect).
        await agen.aclose()

    prompt_tokens = last_response.prompt_tokens if last_response else 0
    completion_tokens = last_response.generation_tokens if last_response else 0
    finish_reason = (
        (getattr(last_response, "finish_reason", None) or "stop") if last_response else "stop"
    )
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
