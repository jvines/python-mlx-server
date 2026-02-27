from dataclasses import dataclass

import mlx.core as mx
import pytest

from mlx_server.managers.embedding import _normalize_embedding_output


@dataclass
class _ModelOutput:
    text_embeds: object | None = None
    pooler_output: object | None = None
    last_hidden_state: object | None = None


def test_normalize_embedding_output_text_embeds():
    out = _ModelOutput(text_embeds=mx.array([[0.1, 0.2, 0.3]]))
    vectors = _normalize_embedding_output(out)
    assert vectors[0] == pytest.approx([0.1, 0.2, 0.3], rel=1e-6, abs=1e-6)


def test_normalize_embedding_output_pooler_output():
    out = _ModelOutput(pooler_output=mx.array([[1.0, 2.0]]))
    vectors = _normalize_embedding_output(out)
    assert vectors[0] == pytest.approx([1.0, 2.0], rel=1e-6, abs=1e-6)


def test_normalize_embedding_output_single_vector_wrapped():
    vectors = _normalize_embedding_output(mx.array([0.5, 0.6]))
    assert vectors[0] == pytest.approx([0.5, 0.6], rel=1e-6, abs=1e-6)
