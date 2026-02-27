"""
/v1/models  — OpenAI-compatible model listing + registration/unload extensions.

Standard OpenAI endpoints:
  GET  /v1/models              — list all registered models
  GET  /v1/models/{model_id}   — get a single model's info

Extensions:
  POST   /v1/models/register   — register a local model path
  POST   /v1/models/{id}/load  — preload a model into memory
  DELETE /v1/models/{id}       — unload from memory (keeps registry entry)
  DELETE /v1/models/{id}/unregister — remove from registry entirely
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator, model_validator

from ..managers import embedding_manager, generative_manager, vlm_manager
from ..registry import ModelType, registry

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mlx-server"
    type: str
    loaded: bool
    path: str


class RegisterRequest(BaseModel):
    id: str
    path: str
    type: ModelType
    mmproj: Optional[str] = None
    overwrite: bool = False

    @field_validator("path")
    @classmethod
    def path_exists(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            raise ValueError("path must be absolute")
        if not p.exists():
            raise ValueError(f"Path does not exist: {v}")
        return str(p.resolve())

    @field_validator("mmproj")
    @classmethod
    def mmproj_exists_if_set(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        p = Path(v)
        if not p.is_absolute():
            raise ValueError("mmproj must be an absolute path")
        if not p.exists():
            raise ValueError(f"mmproj path does not exist: {v}")
        return str(p.resolve())

    @model_validator(mode="after")
    def validate_mmproj_for_model_type(self) -> "RegisterRequest":
        if self.mmproj is not None and self.type != "vlm":
            raise ValueError("mmproj is only valid for models with type='vlm'")
        return self


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _loaded_ids() -> set:
    return set(
        generative_manager.loaded_models()
        + embedding_manager.loaded_models()
        + vlm_manager.loaded_models()
    )


def _to_model_object(model_id: str, loaded_set: set) -> ModelObject:
    entry = registry.get(model_id)
    return ModelObject(
        id=model_id,
        created=entry.created,
        type=entry.type,
        loaded=model_id in loaded_set,
        path=entry.path,
    )


def _unload_from_all(model_id: str) -> bool:
    unloaded = generative_manager.unload(model_id)
    unloaded |= embedding_manager.unload(model_id)
    unloaded |= vlm_manager.unload(model_id)
    return unloaded


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    loaded = _loaded_ids()
    return ModelListResponse(
        data=[_to_model_object(mid, loaded) for mid in registry.list_all()]
    )


@router.get("/v1/models/{model_id}", response_model=ModelObject)
async def get_model(model_id: str):
    entry = registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return _to_model_object(model_id, _loaded_ids())


@router.post("/v1/models/register", response_model=ModelObject)
async def register_model(body: RegisterRequest):
    try:
        registry.register(
            body.id,
            body.path,
            body.type,
            mmproj=body.mmproj,
            overwrite=body.overwrite,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _to_model_object(body.id, _loaded_ids())


@router.post("/v1/models/{model_id}/load", response_model=ModelObject)
async def preload_model(model_id: str):
    entry = registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    try:
        if entry.type == "generative":
            await generative_manager.load_model(model_id, entry.path)
        elif entry.type == "embedding":
            await embedding_manager.load_model(model_id, entry.path)
        elif entry.type == "vlm":
            await vlm_manager.load_model(model_id, entry.path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load model: {exc}")

    return _to_model_object(model_id, _loaded_ids())


@router.delete("/v1/models/{model_id}", response_model=Dict[str, Any])
async def unload_model(model_id: str):
    if registry.get(model_id) is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    unloaded = _unload_from_all(model_id)
    return {"id": model_id, "unloaded": unloaded}


@router.delete("/v1/models/{model_id}/unregister", response_model=Dict[str, Any])
async def unregister_model(model_id: str):
    _unload_from_all(model_id)
    removed = registry.unregister(model_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return {"id": model_id, "unregistered": True}
