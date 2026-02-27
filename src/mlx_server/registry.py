"""
Model registry — maps user-defined model IDs to paths and backend metadata.

Stored as a JSON file at settings.model_registry_path so it persists across
server restarts. Register a model once; reference it forever by its ID.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import settings

logger = logging.getLogger(__name__)

ModelType = Literal["generative", "embedding", "vlm"]


class ModelEntry(BaseModel):
    path: str
    type: ModelType
    created: int = Field(default_factory=lambda: int(time.time()))
    # VLMs may carry a separate multimodal projector file (e.g. mmproj-*.gguf)
    mmproj: Optional[str] = None

    @field_validator("path")
    @classmethod
    def path_must_exist(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            raise ValueError(f"Path must be absolute: {v}")
        if not p.exists():
            raise ValueError(f"Path does not exist: {v}")
        return str(p.resolve())

    @field_validator("mmproj")
    @classmethod
    def mmproj_must_exist_if_set(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        p = Path(v)
        if not p.is_absolute():
            raise ValueError(f"mmproj path must be absolute: {v}")
        if not p.exists():
            raise ValueError(f"mmproj path does not exist: {v}")
        return str(p.resolve())


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, ModelEntry] = {}
        self._path = settings.model_registry_path
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with self._lock:
            if not self._path.exists():
                return
            try:
                raw = json.loads(self._path.read_text())
                for model_id, data in raw.items():
                    try:
                        self._entries[model_id] = ModelEntry(**data)
                    except Exception as exc:
                        logger.warning(f"Skipping invalid registry entry '{model_id}': {exc}")
                logger.info(f"Registry loaded {len(self._entries)} models from {self._path}")
            except Exception as exc:
                logger.error(f"Failed to load registry from {self._path}: {exc}")

    def _save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.model_dump() for k, v in self._entries.items()}
            tmp_path = self._path.parent / f"{self._path.name}.tmp"
            tmp_path.write_text(json.dumps(data, indent=2, default=str))
            tmp_path.replace(self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model_id: str,
        path: str,
        model_type: ModelType,
        mmproj: Optional[str] = None,
        overwrite: bool = False,
    ) -> ModelEntry:
        with self._lock:
            if model_id in self._entries and not overwrite:
                raise ValueError(
                    f"Model '{model_id}' is already registered. "
                    "Pass overwrite=True or choose a different ID."
                )
            entry = ModelEntry(path=path, type=model_type, mmproj=mmproj)
            self._entries[model_id] = entry
            self._save()
            logger.info(f"Registered '{model_id}' ({model_type}) → {entry.path}")
            return entry

    def get(self, model_id: str) -> Optional[ModelEntry]:
        with self._lock:
            return self._entries.get(model_id)

    def list_all(self) -> Dict[str, ModelEntry]:
        with self._lock:
            return dict(self._entries)

    def unregister(self, model_id: str) -> bool:
        with self._lock:
            if model_id not in self._entries:
                return False
            del self._entries[model_id]
            self._save()
            logger.info(f"Unregistered '{model_id}'")
            return True


registry = ModelRegistry()
