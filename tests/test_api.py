"""
Integration tests for the MLX server API.

These tests mock the model managers so no actual models need to be loaded.
Run with: uv run pytest tests/ -v
"""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mlx_server.api.app import app
from mlx_server.registry import ModelEntry


def _entry(path: str, type: str, **kw) -> ModelEntry:
    """Construct a ModelEntry bypassing path-existence validation (tests only)."""
    return ModelEntry.model_construct(path=path, type=type, **kw)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _make_fake_response(text: str, prompt_tokens: int = 10, gen_tokens: int = 5):
    r = MagicMock()
    r.text = text
    r.prompt_tokens = prompt_tokens
    r.generation_tokens = gen_tokens
    return r


async def _fake_stream(*chunks: str) -> AsyncGenerator:
    for chunk in chunks:
        yield _make_fake_response(chunk)


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_list_models_empty(self, client):
        with patch("mlx_server.api.models.registry") as mock_reg:
            mock_reg.list_all.return_value = {}
            response = client.get("/v1/models")
        assert response.status_code == 200
        assert response.json()["object"] == "list"
        assert response.json()["data"] == []

    def test_register_model(self, client, tmp_path):
        model_dir = tmp_path / "mymodel"
        model_dir.mkdir()

        with patch("mlx_server.api.models.registry") as mock_reg:
            entry = _entry(str(model_dir), "generative")
            mock_reg.register.return_value = entry
            mock_reg.get.return_value = entry

            response = client.post(
                "/v1/models/register",
                json={"id": "test-model", "path": str(model_dir), "type": "generative"},
            )
        assert response.status_code == 200
        assert response.json()["id"] == "test-model"

    def test_register_nonexistent_path_rejected(self, client):
        response = client.post(
            "/v1/models/register",
            json={"id": "bad", "path": "/nonexistent/path", "type": "generative"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_unknown_model_returns_404(self, client):
        with patch("mlx_server.api.chat.registry") as mock_reg:
            mock_reg.get.return_value = None
            response = client.post(
                "/v1/chat/completions",
                json={"model": "ghost-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 404

    def test_embedding_model_rejected_for_chat(self, client):
        with patch("mlx_server.api.chat.registry") as mock_reg:
            mock_reg.get.return_value = _entry("/fake", "embedding")
            response = client.post(
                "/v1/chat/completions",
                json={"model": "embed-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 400

    def test_non_streaming_response(self, client):
        entry = _entry("/fake/model", "generative")

        async def fake_gen(*a, **kw):
            yield _make_fake_response("Hello ", 8, 2)
            yield _make_fake_response("world!", 8, 3)

        with (
            patch("mlx_server.api.chat.registry") as mock_reg,
            patch("mlx_server.api.chat.generative_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.stream = fake_gen

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-llm",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello world!"
        assert data["usage"]["completion_tokens"] == 3

    def test_kv_params_accepted(self, client):
        entry = _entry("/fake/model", "generative")

        async def fake_gen(*a, **kw):
            assert kw["max_kv_size"] == 4096
            assert kw["kv_bits"] == 4
            yield _make_fake_response("ok", 5, 1)

        with (
            patch("mlx_server.api.chat.registry") as mock_reg,
            patch("mlx_server.api.chat.generative_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.stream = fake_gen

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_kv_size": 4096,
                    "kv_bits": 4,
                    "stream": False,
                },
            )
        assert response.status_code == 200

    def test_temperature_out_of_range_rejected(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 5.0,
            },
        )
        assert response.status_code == 422

    def test_empty_messages_rejected(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": []},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_unknown_model_returns_404(self, client):
        with patch("mlx_server.api.embeddings.registry") as mock_reg:
            mock_reg.get.return_value = None
            response = client.post(
                "/v1/embeddings",
                json={"model": "ghost", "input": "hello"},
            )
        assert response.status_code == 404

    def test_generative_model_rejected(self, client):
        with patch("mlx_server.api.embeddings.registry") as mock_reg:
            mock_reg.get.return_value = _entry("/fake", "generative")
            response = client.post(
                "/v1/embeddings",
                json={"model": "llm", "input": "hello"},
            )
        assert response.status_code == 400

    def test_single_string_input(self, client):
        entry = _entry("/fake/embed", "embedding")
        fake_vector = [0.1, 0.2, 0.3]

        with (
            patch("mlx_server.api.embeddings.registry") as mock_reg,
            patch("mlx_server.api.embeddings.embedding_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.embed = AsyncMock(return_value=[fake_vector])

            response = client.post(
                "/v1/embeddings",
                json={"model": "embed-model", "input": "hello world"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"][0]["embedding"] == fake_vector
        assert data["data"][0]["index"] == 0

    def test_batch_input(self, client):
        entry = _entry("/fake/embed", "embedding")
        fake_vectors = [[0.1, 0.2], [0.3, 0.4]]

        with (
            patch("mlx_server.api.embeddings.registry") as mock_reg,
            patch("mlx_server.api.embeddings.embedding_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.embed = AsyncMock(return_value=fake_vectors)

            response = client.post(
                "/v1/embeddings",
                json={"model": "embed-model", "input": ["text one", "text two"]},
            )

        assert response.status_code == 200
        assert len(response.json()["data"]) == 2
