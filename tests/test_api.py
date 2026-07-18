"""
Integration tests for the MLX server API.

These tests mock the model managers so no actual models need to be loaded.
Run with: uv run pytest tests/ -v
"""

from __future__ import annotations

import time
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mlx_server.api.app import app
from mlx_server.registry import ModelEntry


def _entry(path: str, type: str, **kw) -> ModelEntry:
    """Construct a ModelEntry bypassing path-existence validation (tests only)."""
    return ModelEntry.model_construct(path=path, type=type, created=int(time.time()), **kw)


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
    r.generation_tps = 50.0
    r.finish_reason = "stop"
    return r


async def _fake_stream(*chunks: str) -> AsyncGenerator:
    for chunk in chunks:
        yield _make_fake_response(chunk)


def _parse_sse(text: str) -> list:
    """Parse SSE 'data: {json}' frames, dropping the [DONE] sentinel."""
    import json as _json

    frames = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                continue
            frames.append(_json.loads(payload))
    return frames


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

    def test_register_relative_path_rejected(self, client):
        response = client.post(
            "/v1/models/register",
            json={"id": "bad", "path": "relative/model", "type": "generative"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_unknown_model_returns_404(self, client):
        with patch("mlx_server.api.deps.registry") as mock_reg:
            mock_reg.get.return_value = None
            response = client.post(
                "/v1/chat/completions",
                json={"model": "ghost-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 404

    def test_embedding_model_rejected_for_chat(self, client):
        with patch("mlx_server.api.deps.registry") as mock_reg:
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
            patch("mlx_server.api.deps.registry") as mock_reg,
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
            patch("mlx_server.api.deps.registry") as mock_reg,
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

    def test_enable_thinking_forwarded(self, client):
        entry = _entry("/fake/model", "generative")

        async def fake_gen(*a, **kw):
            assert kw["enable_thinking"] is False
            yield _make_fake_response("ok", 5, 1)

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
            patch("mlx_server.api.chat.generative_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.stream = fake_gen

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "enable_thinking": False,
                    "stream": False,
                },
            )
        assert response.status_code == 200

    def test_vlm_chat_does_not_forward_top_p(self, client):
        """Regression: the real VLMModelManager.stream() has no top_p param, so
        chat.py must strip it. Previously every VLM request 500'd with
        TypeError: stream() got an unexpected keyword argument 'top_p'."""
        entry = _entry("/fake/vlm", "vlm")
        captured: dict = {}

        async def fake_vlm_stream(model, path, messages, images=None, **kw):
            captured.update(kw)
            yield _make_fake_response("hi", 5, 1)

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
            patch("mlx_server.api.chat.vlm_manager") as mock_vlm,
        ):
            mock_reg.get.return_value = entry
            mock_vlm.stream = fake_vlm_stream

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-vlm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "top_p": 0.5,
                    "max_tokens": 8,
                    "stream": False,
                },
            )
        assert response.status_code == 200
        assert "top_p" not in captured
        # Other sampling/KV kwargs are still forwarded.
        assert captured["max_tokens"] == 8
        assert "temperature" in captured

    def test_vlm_response_without_finish_reason(self, client):
        """Regression: mlx_vlm's GenerationResult has no finish_reason attribute
        (unlike mlx_lm's). Response assembly must not AttributeError on it —
        this was masked until the top_p crash was fixed."""
        entry = _entry("/fake/vlm", "vlm")

        class _VLMResult:  # mimics mlx_vlm.GenerationResult (no finish_reason)
            text = "hi there"
            prompt_tokens = 5
            generation_tokens = 2
            generation_tps = 40.0

        async def fake_vlm_stream(model, path, messages, images=None, **kw):
            yield _VLMResult()

        for stream in (False, True):
            with (
                patch("mlx_server.api.deps.registry") as mock_reg,
                patch("mlx_server.api.chat.vlm_manager") as mock_vlm,
            ):
                mock_reg.get.return_value = entry
                mock_vlm.stream = fake_vlm_stream
                r = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-vlm",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": stream,
                    },
                )
            assert r.status_code == 200, f"stream={stream}"
            if stream:
                frames = _parse_sse(r.text)
                assert frames[-1]["choices"][0]["finish_reason"] == "stop"
            else:
                assert r.json()["choices"][0]["message"]["content"] == "hi there"
                assert r.json()["choices"][0]["finish_reason"] == "stop"

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

    def test_temperature_zero_is_preserved(self, client):
        entry = _entry("/fake/model", "generative")

        async def fake_gen(*a, **kw):
            assert kw["temperature"] == 0.0
            yield _make_fake_response("ok", 5, 1)

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
            patch("mlx_server.api.chat.generative_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.stream = fake_gen

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.0,
                    "stream": False,
                },
            )
        assert response.status_code == 200

    def test_empty_messages_rejected(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": []},
        )
        assert response.status_code == 422

    def test_n_greater_than_one_rejected(self, client):
        entry = _entry("/fake/model", "generative")
        with patch("mlx_server.api.deps.registry") as mock_reg:
            mock_reg.get.return_value = entry
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "n": 2,
                },
            )
        assert response.status_code == 400

    def test_streaming_response_frames(self, client):
        """The SSE stream must emit a role delta, content deltas, a closing
        chunk with usage, and the [DONE] sentinel."""
        entry = _entry("/fake/model", "generative")

        async def fake_gen(*a, **kw):
            yield _make_fake_response("Hello ", 8, 1)
            yield _make_fake_response("world", 8, 2)

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
            patch("mlx_server.api.chat.generative_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.stream = fake_gen
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _parse_sse(response.text)
        assert frames[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        content = "".join(
            f["choices"][0]["delta"].get("content", "") for f in frames
        )
        assert content == "Hello world"
        assert frames[-1]["choices"][0]["finish_reason"] == "stop"
        assert frames[-1]["usage"]["completion_tokens"] == 2
        assert response.text.strip().endswith("data: [DONE]")

    def test_streaming_midstream_error_emits_error_event(self, client):
        """Regression (n6): a failure after headers are committed must emit an
        error event + [DONE], not just truncate the stream."""
        entry = _entry("/fake/model", "generative")

        async def fake_gen(*a, **kw):
            yield _make_fake_response("partial", 8, 1)
            raise RuntimeError("boom mid-stream")

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
            patch("mlx_server.api.chat.generative_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.stream = fake_gen
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        frames = _parse_sse(response.text)
        assert any("error" in f for f in frames)
        assert response.text.strip().endswith("data: [DONE]")

    def test_streaming_startup_error_returns_503(self, client):
        """Regression (n6): a load/setup failure before the first token must be
        a proper HTTP error, not a 200 with an empty body."""
        entry = _entry("/fake/model", "generative")

        async def fake_gen(*a, **kw):
            raise RuntimeError("model load failed")
            yield  # pragma: no cover - makes this an async generator

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
            patch("mlx_server.api.chat.generative_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.stream = fake_gen
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert response.status_code == 503

    def test_blocking_generation_error_returns_500(self, client):
        """Regression (n8): non-streaming generation failure must be a 500, not
        an unhandled exception."""
        entry = _entry("/fake/model", "generative")

        async def fake_gen(*a, **kw):
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
            patch("mlx_server.api.chat.generative_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.stream = fake_gen
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_unknown_model_returns_404(self, client):
        with patch("mlx_server.api.deps.registry") as mock_reg:
            mock_reg.get.return_value = None
            response = client.post(
                "/v1/embeddings",
                json={"model": "ghost", "input": "hello"},
            )
        assert response.status_code == 404

    def test_generative_model_rejected(self, client):
        with patch("mlx_server.api.deps.registry") as mock_reg:
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
            patch("mlx_server.api.deps.registry") as mock_reg,
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
        mock_mgr.embed.assert_awaited_once_with("embed-model", "/fake/embed", "hello world")

    def test_batch_input(self, client):
        entry = _entry("/fake/embed", "embedding")
        fake_vectors = [[0.1, 0.2], [0.3, 0.4]]

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
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
        mock_mgr.embed.assert_awaited_once_with(
            "embed-model", "/fake/embed", ["text one", "text two"]
        )

    def test_base64_encoding_format(self, client):
        """OpenAI SDK defaults to base64 — it must round-trip to the same floats."""
        import base64 as _b64
        import struct

        entry = _entry("/fake/embed", "embedding")
        fake_vector = [0.1, 0.2, 0.3]

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
            patch("mlx_server.api.embeddings.embedding_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.embed = AsyncMock(return_value=[fake_vector])

            response = client.post(
                "/v1/embeddings",
                json={"model": "embed-model", "input": "hello", "encoding_format": "base64"},
            )

        assert response.status_code == 200
        emb = response.json()["data"][0]["embedding"]
        assert isinstance(emb, str)
        decoded = list(struct.unpack("<3f", _b64.b64decode(emb)))
        assert decoded == pytest.approx(fake_vector, abs=1e-6)

    def test_dimensions_accepted_and_ignored(self, client):
        entry = _entry("/fake/embed", "embedding")

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
            patch("mlx_server.api.embeddings.embedding_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.embed = AsyncMock(return_value=[[0.1, 0.2]])

            response = client.post(
                "/v1/embeddings",
                json={"model": "embed-model", "input": "hello", "dimensions": 256},
            )

        assert response.status_code == 200
        assert response.json()["data"][0]["embedding"] == [0.1, 0.2]

    def test_input_too_many_items_rejected(self, client):
        response = client.post(
            "/v1/embeddings",
            json={"model": "embed-model", "input": ["x"] * 2049},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Request-logging middleware
# ---------------------------------------------------------------------------


class TestMiddleware:
    def test_generates_request_id(self, client):
        with patch("mlx_server.api.models.registry") as mock_reg:
            mock_reg.list_all.return_value = []
            r = client.get("/v1/models")
        assert r.status_code == 200
        assert r.headers.get("x-request-id")

    def test_echoes_client_request_id(self, client):
        with patch("mlx_server.api.models.registry") as mock_reg:
            mock_reg.list_all.return_value = []
            r = client.get("/v1/models", headers={"X-Request-ID": "abc123"})
        assert r.headers.get("x-request-id") == "abc123"

    def test_request_id_set_on_streaming_response(self, client):
        entry = _entry("/fake/model", "generative")

        async def fake_gen(*a, **kw):
            yield _make_fake_response("hi", 8, 1)

        with (
            patch("mlx_server.api.deps.registry") as mock_reg,
            patch("mlx_server.api.chat.generative_manager") as mock_mgr,
        ):
            mock_reg.get.return_value = entry
            mock_mgr.stream = fake_gen
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert r.headers.get("x-request-id")


# ---------------------------------------------------------------------------
# /v1/models management endpoints
# ---------------------------------------------------------------------------


class TestModelManagement:
    def test_get_model_found(self, client):
        with patch("mlx_server.api.models.registry") as mock_reg:
            mock_reg.get.return_value = _entry("/fake/m", "generative")
            r = client.get("/v1/models/m")
        assert r.status_code == 200
        assert r.json()["id"] == "m"
        assert r.json()["loaded"] is False

    def test_get_model_not_found(self, client):
        with patch("mlx_server.api.models.registry") as mock_reg:
            mock_reg.get.return_value = None
            r = client.get("/v1/models/ghost")
        assert r.status_code == 404

    def test_load_model_success(self, client):
        entry = _entry("/fake/m", "generative")
        with (
            patch("mlx_server.api.models.registry") as mock_reg,
            patch("mlx_server.api.models.generative_manager") as mock_gen,
        ):
            mock_reg.get.return_value = entry
            mock_gen.load_model = AsyncMock(return_value=(object(), object()))
            mock_gen.loaded_models.return_value = ["m"]
            r = client.post("/v1/models/m/load")
        assert r.status_code == 200
        assert r.json()["loaded"] is True
        mock_gen.load_model.assert_awaited_once()

    def test_load_model_failure_returns_503(self, client):
        entry = _entry("/fake/m", "generative")
        with (
            patch("mlx_server.api.models.registry") as mock_reg,
            patch("mlx_server.api.models.generative_manager") as mock_gen,
        ):
            mock_reg.get.return_value = entry
            mock_gen.load_model = AsyncMock(side_effect=RuntimeError("boom"))
            r = client.post("/v1/models/m/load")
        assert r.status_code == 503

    def test_load_unknown_model_404(self, client):
        with patch("mlx_server.api.models.registry") as mock_reg:
            mock_reg.get.return_value = None
            r = client.post("/v1/models/ghost/load")
        assert r.status_code == 404

    def test_unload_model_not_loaded(self, client):
        with patch("mlx_server.api.models.registry") as mock_reg:
            mock_reg.get.return_value = _entry("/fake/m", "generative")
            r = client.delete("/v1/models/m")
        assert r.status_code == 200
        assert r.json()["unloaded"] is False

    def test_unload_unknown_model_404(self, client):
        with patch("mlx_server.api.models.registry") as mock_reg:
            mock_reg.get.return_value = None
            r = client.delete("/v1/models/ghost")
        assert r.status_code == 404

    def test_unregister_model(self, client):
        with patch("mlx_server.api.models.registry") as mock_reg:
            mock_reg.unregister.return_value = True
            r = client.delete("/v1/models/m/unregister")
        assert r.status_code == 200
        assert r.json()["unregistered"] is True

    def test_unregister_unknown_model_404(self, client):
        with patch("mlx_server.api.models.registry") as mock_reg:
            mock_reg.unregister.return_value = False
            r = client.delete("/v1/models/ghost/unregister")
        assert r.status_code == 404
