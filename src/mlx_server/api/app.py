import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

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

app.include_router(chat_router)
app.include_router(embed_router)
app.include_router(models_router)
app.include_router(convert_router)
