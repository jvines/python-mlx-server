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
_DEFAULT_MAX_JOBS = 200
_DEFAULT_RETENTION_SECONDS = 24 * 60 * 60


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
    def __init__(
        self,
        *,
        max_jobs: int = _DEFAULT_MAX_JOBS,
        retention_seconds: int = _DEFAULT_RETENTION_SECONDS,
    ) -> None:
        self._jobs: Dict[str, ConversionJob] = {}
        self._jobs_lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.Lock()
        self._max_jobs = max_jobs
        self._retention_seconds = retention_seconds

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="mlx_conv"
                )
            return self._executor

    def get(self, job_id: str) -> Optional[ConversionJob]:
        with self._jobs_lock:
            self._prune_locked()
            return self._jobs.get(job_id)

    def list_all(self) -> List[ConversionJob]:
        with self._jobs_lock:
            self._prune_locked()
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _prune_locked(self) -> None:
        now = time.time()
        if self._retention_seconds > 0:
            for job_id in list(self._jobs):
                job = self._jobs[job_id]
                if (
                    job.completed_at is not None
                    and now - job.completed_at > self._retention_seconds
                ):
                    del self._jobs[job_id]

        overflow = len(self._jobs) - self._max_jobs
        if overflow <= 0:
            return

        removable = sorted(
            (
                j for j in self._jobs.values()
                if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            ),
            key=lambda j: j.created_at,
        )
        for job in removable[:overflow]:
            self._jobs.pop(job.id, None)

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
        with self._jobs_lock:
            self._jobs[job_id] = job
            self._prune_locked()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        def _run() -> None:
            with self._jobs_lock:
                job.status = JobStatus.RUNNING
            try:
                worker_fn(job)
                with self._jobs_lock:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = time.time()
            except Exception as exc:
                with self._jobs_lock:
                    job.status = JobStatus.FAILED
                    job.error = str(exc)
                    job.completed_at = time.time()
                # Remove partial output so a retry doesn't hit FileExistsError
                out = Path(job.output_path)
                if out.exists():
                    logger.warning("Job %s failed — removing partial output: %s", job.id, out)
                    if out.is_dir():
                        shutil.rmtree(out, ignore_errors=True)
                    else:
                        out.unlink(missing_ok=True)

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
