import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from ..config import settings
from ..conversion.jobs import job_manager
from ..managers import generative_manager, embedding_manager, vlm_manager
from .chat import router as chat_router
from .convert import router as convert_router
from .embeddings import router as embed_router
from .models import router as models_router

logger = logging.getLogger(__name__)

_CHECK_INTERVAL = 60  # seconds between eviction sweeps


async def _eviction_loop() -> None:
    ttl = settings.model_ttl_seconds
    if ttl <= 0:
        return
    while True:
        await asyncio.sleep(_CHECK_INTERVAL)
        for manager in (generative_manager, embedding_manager, vlm_manager):
            evicted = manager.evict_stale(ttl)
            for mid in evicted:
                logger.info("TTL eviction: '%s' evicted after %ds idle", mid, ttl)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_eviction_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        job_manager.shutdown()


app = FastAPI(
    title="MLX Model Server",
    version="0.2.0",
    description="OpenAI-compatible inference server for MLX models on Apple Silicon.",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
    started = time.perf_counter()
    logger.info(
        "request_in id=%s method=%s path=%s client=%s",
        request_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "-",
    )
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.exception(
            "request_err id=%s method=%s path=%s elapsed_ms=%d",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    response.headers["X-Request-ID"] = request_id
    headers_ms = int((time.perf_counter() - started) * 1000)

    # Under Starlette's BaseHTTPMiddleware even a fully-buffered response arrives
    # here wrapped with a body_iterator, so detect a genuine stream by its
    # content type rather than by the iterator's presence.
    is_stream = response.headers.get("content-type", "").startswith("text/event-stream")
    body_iterator = getattr(response, "body_iterator", None)
    if not is_stream or body_iterator is None:
        # Buffered response: status and timing are final now.
        logger.info(
            "request_out id=%s method=%s path=%s status=%d elapsed_ms=%d",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            headers_ms,
        )
        return response

    # Streaming response: the body hasn't been sent yet, so headers_ms is only
    # time-to-first-byte. Wrap the iterator to log the terminal outcome (and any
    # mid-stream failure) once the stream actually finishes.
    logger.info(
        "request_stream id=%s method=%s path=%s status=%d ttfb_ms=%d",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        headers_ms,
    )

    async def _logged_body():
        try:
            async for chunk in body_iterator:
                yield chunk
        except Exception:
            total_ms = int((time.perf_counter() - started) * 1000)
            logger.exception(
                "request_stream_err id=%s method=%s path=%s elapsed_ms=%d",
                request_id,
                request.method,
                request.url.path,
                total_ms,
            )
            raise
        else:
            total_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "request_stream_end id=%s method=%s path=%s status=%d elapsed_ms=%d",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                total_ms,
            )

    response.body_iterator = _logged_body()
    return response


app.include_router(chat_router)
app.include_router(embed_router)
app.include_router(models_router)
app.include_router(convert_router)
