"""
Conversion endpoints.

POST /v1/convert/hf    — Download from HuggingFace and convert to MLX
POST /v1/convert/gguf  — Convert a local GGUF file to MLX

GET  /v1/convert/{job_id}         — Poll job status
GET  /v1/convert/{job_id}/stream  — SSE stream of progress messages
GET  /v1/convert                   — List all jobs
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..conversion import job_manager, JobStatus
from ..conversion.hf import convert_from_hf
from ..conversion.gguf import convert_from_gguf
from ..registry import registry

router = APIRouter(prefix="/v1/convert", tags=["convert"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class HFConvertRequest(BaseModel):
    hf_repo: str = Field(..., description="HuggingFace repo ID, e.g. 'Qwen/Qwen3-8B'")
    output_path: str = Field(..., description="Absolute local path for the converted model")
    model_type: Literal["generative", "embedding", "vlm"] = "generative"
    quantize: bool = True
    q_bits: Literal[4, 8] = 4
    q_group_size: int = Field(default=64, ge=1)
    dtype: Optional[str] = None
    register_as: Optional[str] = Field(
        default=None,
        description="If set, register the converted model under this ID after completion",
    )


class GGUFConvertRequest(BaseModel):
    gguf_path: str = Field(..., description="Absolute path to the local .gguf file")
    hf_tokenizer_repo: str = Field(
        ...,
        description=(
            "HuggingFace repo ID used to download the tokenizer "
            "(e.g. 'meta-llama/Llama-3.3-70B-Instruct')"
        ),
    )
    output_path: str = Field(..., description="Absolute local path for the converted model")
    model_type: Literal["generative"] = "generative"
    quantize: bool = True
    q_bits: Literal[4, 8] = 4
    q_group_size: int = Field(default=64, ge=1)
    register_as: Optional[str] = Field(
        default=None,
        description="If set, register the converted model under this ID after completion",
    )

    @field_validator("gguf_path")
    @classmethod
    def path_must_end_with_gguf(cls, v: str) -> str:
        if not v.lower().endswith(".gguf"):
            raise ValueError("gguf_path must end with .gguf")
        return v


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


def _job_response(job) -> dict:
    return {
        "job_id": job.id,
        "source": job.source,
        "model_type": job.model_type,
        "output_path": job.output_path,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "register_as": job.register_as,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/hf", status_code=202)
async def convert_from_hf_endpoint(request: HFConvertRequest):
    """Submit an async HuggingFace → MLX conversion job."""

    def worker(job):
        convert_from_hf(
            job=job,
            hf_repo=request.hf_repo,
            quantize=request.quantize,
            q_bits=request.q_bits,
            q_group_size=request.q_group_size,
            dtype=request.dtype,
        )
        if job.register_as:
            registry.register(job.register_as, job.output_path, job.model_type, overwrite=True)

    job = job_manager.submit(
        source="hf",
        model_type=request.model_type,
        output_path=request.output_path,
        worker_fn=worker,
        register_as=request.register_as,
    )
    return _job_response(job)


@router.post("/gguf", status_code=202)
async def convert_from_gguf_endpoint(request: GGUFConvertRequest):
    """Submit an async GGUF → MLX conversion job."""

    def worker(job):
        convert_from_gguf(
            job=job,
            gguf_path=request.gguf_path,
            hf_tokenizer_repo=request.hf_tokenizer_repo,
            quantize=request.quantize,
            q_bits=request.q_bits,
            q_group_size=request.q_group_size,
        )
        if job.register_as:
            registry.register(job.register_as, job.output_path, job.model_type, overwrite=True)

    job = job_manager.submit(
        source="gguf",
        model_type=request.model_type,
        output_path=request.output_path,
        worker_fn=worker,
        register_as=request.register_as,
    )
    return _job_response(job)


@router.get("")
async def list_jobs():
    """List all conversion jobs (most recent first)."""
    return [_job_response(j) for j in job_manager.list_all()]


@router.get("/{job_id}")
async def get_job(job_id: str):
    """Poll a conversion job's current status."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _job_response(job)


@router.get("/{job_id}/stream")
async def stream_job(job_id: str):
    """
    SSE stream that emits progress events until the job completes or fails.
    Each event is a JSON object with the same shape as GET /{job_id}.
    """
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    async def _generate():
        last_progress = None
        while True:
            current = job.progress
            if current != last_progress:
                last_progress = current
                payload = json.dumps(_job_response(job))
                yield f"data: {payload}\n\n"

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                # Emit one final event so the client sees terminal state
                payload = json.dumps(_job_response(job))
                yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
