"""
Conversion job tracking.

Jobs run in a thread pool and report progress via a shared state dict.
Clients poll GET /v1/convert/{job_id} or stream SSE from /v1/convert/{job_id}/stream.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ConversionJob:
    id: str
    source: str          # "hf" or "gguf"
    model_type: str      # "generative" | "embedding" | "vlm"
    output_path: str
    status: JobStatus = JobStatus.QUEUED
    progress: str = "Queued"
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    register_as: Optional[str] = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, ConversionJob] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.Lock()

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="mlx_conv"
                )
            return self._executor

    def get(self, job_id: str) -> Optional[ConversionJob]:
        return self._jobs.get(job_id)

    def list_all(self) -> List[ConversionJob]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def submit(
        self,
        source: str,
        model_type: str,
        output_path: str,
        worker_fn: Callable[[ConversionJob], None],
        register_as: Optional[str] = None,
    ) -> ConversionJob:
        job_id = f"conv-{uuid.uuid4().hex[:8]}"
        job = ConversionJob(
            id=job_id,
            source=source,
            model_type=model_type,
            output_path=output_path,
            register_as=register_as,
        )
        self._jobs[job_id] = job

        loop = asyncio.get_event_loop()

        def _run() -> None:
            job.status = JobStatus.RUNNING
            try:
                worker_fn(job)
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.completed_at = time.time()
                # Remove partial output so a retry doesn't hit FileExistsError
                out = Path(job.output_path)
                if out.exists():
                    logger.warning("Job %s failed — removing partial output: %s", job.id, out)
                    shutil.rmtree(out, ignore_errors=True)

        loop.run_in_executor(self._get_executor(), _run)
        return job

    def shutdown(self) -> None:
        """Cancel queued futures and stop accepting new work. Running jobs finish."""
        with self._executor_lock:
            if self._executor is None:
                return
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


job_manager = JobManager()
