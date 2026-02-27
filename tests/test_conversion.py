"""
Tests for the conversion subsystem.

All heavy work (mx.load, HF downloads, mlx_lm.convert) is mocked so
these tests run without real models or network access.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient

from mlx_server.api.app import app
from mlx_server.conversion.jobs import ConversionJob, JobStatus, JobManager


client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Job manager unit tests
# ---------------------------------------------------------------------------


def test_job_manager_submit_and_get():
    mgr = JobManager()
    sentinel = {}

    def worker(job: ConversionJob):
        sentinel["ran"] = True
        job.progress = "done"

    job = mgr.submit(
        source="hf",
        model_type="generative",
        output_path="/tmp/model_out",
        worker_fn=worker,
        register_as=None,
    )

    assert job.id.startswith("conv-")
    assert job.source == "hf"
    assert job.model_type == "generative"
    assert mgr.get(job.id) is job
    assert mgr.get("nonexistent") is None


def test_job_manager_list_all_sorted():
    import time
    mgr = JobManager()

    def noop(job):
        pass

    j1 = mgr.submit("hf", "generative", "/tmp/a", noop)
    time.sleep(0.01)
    j2 = mgr.submit("gguf", "generative", "/tmp/b", noop)

    listed = mgr.list_all()
    # Most recent first
    assert listed[0].id == j2.id
    assert listed[1].id == j1.id


def test_job_manager_prunes_old_completed_jobs():
    import time

    mgr = JobManager(max_jobs=100, retention_seconds=1)

    old = ConversionJob(
        id="conv-old",
        source="hf",
        model_type="generative",
        output_path="/tmp/old",
        status=JobStatus.COMPLETED,
        completed_at=time.time() - 3600,
    )
    fresh = ConversionJob(
        id="conv-fresh",
        source="hf",
        model_type="generative",
        output_path="/tmp/fresh",
        status=JobStatus.COMPLETED,
        completed_at=time.time(),
    )
    mgr._jobs[old.id] = old
    mgr._jobs[fresh.id] = fresh

    listed = mgr.list_all()
    ids = [j.id for j in listed]
    assert "conv-old" not in ids
    assert "conv-fresh" in ids


# ---------------------------------------------------------------------------
# GGUF weight remapping unit tests
# ---------------------------------------------------------------------------


def test_remap_global_weights():
    from mlx_server.conversion.gguf import _remap_weights
    import mlx.core as mx

    weights = {
        "token_embd.weight": mx.zeros((32000, 4096)),
        "output_norm.weight": mx.zeros((4096,)),
        "output.weight": mx.zeros((32000, 4096)),
    }
    remapped = _remap_weights(weights, arch="llama")

    assert "model.embed_tokens.weight" in remapped
    assert "model.norm.weight" in remapped
    assert "lm_head.weight" in remapped
    assert "token_embd.weight" not in remapped


def test_remap_layer_weights():
    from mlx_server.conversion.gguf import _remap_weights
    import mlx.core as mx

    weights = {
        "blk.0.attn_norm.weight":   mx.zeros((4096,)),
        "blk.0.attn_q.weight":      mx.zeros((4096, 4096)),
        "blk.0.attn_k.weight":      mx.zeros((1024, 4096)),
        "blk.0.attn_v.weight":      mx.zeros((1024, 4096)),
        "blk.0.attn_output.weight": mx.zeros((4096, 4096)),
        "blk.0.ffn_gate.weight":    mx.zeros((14336, 4096)),
        "blk.0.ffn_up.weight":      mx.zeros((14336, 4096)),
        "blk.0.ffn_down.weight":    mx.zeros((4096, 14336)),
        "blk.0.ffn_norm.weight":    mx.zeros((4096,)),
    }
    remapped = _remap_weights(weights, arch="llama")

    assert "model.layers.0.input_layernorm.weight" in remapped
    assert "model.layers.0.self_attn.q_proj.weight" in remapped
    assert "model.layers.0.self_attn.k_proj.weight" in remapped
    assert "model.layers.0.self_attn.v_proj.weight" in remapped
    assert "model.layers.0.self_attn.o_proj.weight" in remapped
    assert "model.layers.0.mlp.gate_proj.weight" in remapped
    assert "model.layers.0.mlp.up_proj.weight" in remapped
    assert "model.layers.0.mlp.down_proj.weight" in remapped
    assert "model.layers.0.post_attention_layernorm.weight" in remapped


def test_remap_quantized_weights_preserves_scales():
    """Q8_0 weights come with .scales and .biases — they must be remapped too."""
    from mlx_server.conversion.gguf import _remap_weights
    import mlx.core as mx

    weights = {
        "blk.0.attn_q.weight": mx.zeros((4096, 1024)),
        "blk.0.attn_q.scales": mx.zeros((4096, 64)),
        "blk.0.attn_q.biases": mx.zeros((4096, 64)),
    }
    remapped = _remap_weights(weights, arch="qwen2")

    assert "model.layers.0.self_attn.q_proj.weight" in remapped
    assert "model.layers.0.self_attn.q_proj.scales" in remapped
    assert "model.layers.0.self_attn.q_proj.biases" in remapped


def test_is_already_quantized():
    from mlx_server.conversion.gguf import _is_already_quantized
    import mlx.core as mx

    assert _is_already_quantized({"foo.scales": mx.zeros((1,))})
    assert not _is_already_quantized({"foo.weight": mx.zeros((4, 4))})


def test_build_config_llama():
    from mlx_server.conversion.gguf import _build_config
    import mlx.core as mx

    meta = {
        "general.architecture": "llama",
        "llama.embedding_length": 4096,
        "llama.block_count": 32,
        "llama.feed_forward_length": 14336,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
        "llama.context_length": 131072,
        "llama.rope.freq_base": 500000.0,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
    }
    weights = {"model.embed_tokens.weight": mx.zeros((32000, 4096))}
    config = _build_config("llama", meta, weights)

    assert config["model_type"] == "llama"
    assert config["hidden_size"] == 4096
    assert config["num_hidden_layers"] == 32
    assert config["num_key_value_heads"] == 8
    assert config["rope_theta"] == 500000.0
    assert "vocab_size" in config


# ---------------------------------------------------------------------------
# Conversion API endpoint tests
# ---------------------------------------------------------------------------


def _make_fake_job(source="hf", status=JobStatus.QUEUED):
    from mlx_server.conversion.jobs import ConversionJob
    import time
    return ConversionJob(
        id="conv-abc12345",
        source=source,
        model_type="generative",
        output_path="/tmp/out",
        status=status,
        progress="Queued",
        created_at=time.time(),
    )


def test_list_jobs_empty():
    with patch("mlx_server.api.convert.job_manager") as mock_jm:
        mock_jm.list_all.return_value = []
        resp = client.get("/v1/convert")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_jobs_returns_jobs():
    job = _make_fake_job()
    with patch("mlx_server.api.convert.job_manager") as mock_jm:
        mock_jm.list_all.return_value = [job]
        resp = client.get("/v1/convert")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["job_id"] == "conv-abc12345"


def test_get_job_not_found():
    with patch("mlx_server.api.convert.job_manager") as mock_jm:
        mock_jm.get.return_value = None
        resp = client.get("/v1/convert/conv-notexist")
    assert resp.status_code == 404


def test_get_job_found():
    job = _make_fake_job()
    with patch("mlx_server.api.convert.job_manager") as mock_jm:
        mock_jm.get.return_value = job
        resp = client.get("/v1/convert/conv-abc12345")
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "conv-abc12345"


def test_post_hf_convert_submits_job():
    job = _make_fake_job(source="hf")
    with patch("mlx_server.api.convert.job_manager") as mock_jm:
        mock_jm.submit.return_value = job
        resp = client.post(
            "/v1/convert/hf",
            json={
                "hf_repo": "Qwen/Qwen3-8B",
                "output_path": "/tmp/qwen3-mlx",
                "model_type": "generative",
            },
        )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "conv-abc12345"
    assert resp.json()["source"] == "hf"


def test_post_gguf_convert_submits_job(tmp_path: Path):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")

    job = _make_fake_job(source="gguf")
    with patch("mlx_server.api.convert.job_manager") as mock_jm:
        mock_jm.submit.return_value = job
        resp = client.post(
            "/v1/convert/gguf",
            json={
                "gguf_path": str(gguf_path),
                "hf_tokenizer_repo": "meta-llama/Llama-3.3-70B-Instruct",
                "output_path": "/tmp/llama-mlx",
            },
        )
    assert resp.status_code == 202
    assert resp.json()["source"] == "gguf"


def test_post_hf_convert_rejects_relative_output_path():
    resp = client.post(
        "/v1/convert/hf",
        json={
            "hf_repo": "Qwen/Qwen3-8B",
            "output_path": "relative/out",
            "model_type": "generative",
        },
    )
    assert resp.status_code == 422


def test_post_gguf_convert_rejects_prefer_hf_without_repo(tmp_path: Path):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")

    resp = client.post(
        "/v1/convert/gguf",
        json={
            "gguf_path": str(gguf_path),
            "output_path": "/tmp/out",
            "prefer_hf_tokenizer": True,
        },
    )
    assert resp.status_code == 422


def test_post_gguf_convert_rejects_non_gguf_path():
    resp = client.post(
        "/v1/convert/gguf",
        json={
            "gguf_path": "/Volumes/nas/model.bin",
            "hf_tokenizer_repo": "meta-llama/Llama-3.3-70B-Instruct",
            "output_path": "/tmp/out",
        },
    )
    assert resp.status_code == 422


def test_convert_from_gguf_prefers_gguf_tokenizer_by_default(tmp_path: Path):
    """GGUF extraction is the default tokenizer source when it succeeds."""
    from mlx_server.conversion.gguf import convert_from_gguf

    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    out_dir = tmp_path / "out"
    job = ConversionJob(
        id="conv-testtok1",
        source="gguf",
        model_type="generative",
        output_path=str(out_dir),
    )

    with (
        patch("mlx_server.conversion.gguf.mx.load") as mock_load,
        patch("mlx_server.conversion.gguf._remap_weights", return_value={}),
        patch("mlx_server.conversion.gguf._is_already_quantized", return_value=False),
        patch("mlx_server.conversion.gguf._build_config", return_value={"model_type": "mistral"}),
        patch("mlx_server.conversion.gguf._save_weights"),
        patch("mlx_server.conversion.gguf._download_tokenizer", return_value=["tokenizer.json"]) as mock_download,
        patch("mlx_server.conversion.gguf._extract_tokenizer_from_gguf", return_value=True) as mock_extract,
    ):
        mock_load.return_value = ({}, {"general.architecture": "mistral"})
        convert_from_gguf(
            job=job,
            gguf_path=str(gguf_path),
            hf_tokenizer_repo="mistralai/Mistral-Nemo-Instruct-2407",
            quantize=False,
        )

    mock_extract.assert_called_once()
    mock_download.assert_not_called()


def test_convert_from_gguf_force_hf_tokenizer_when_requested(tmp_path: Path):
    """prefer_hf_tokenizer overrides GGUF extraction when a repo is provided."""
    from mlx_server.conversion.gguf import convert_from_gguf

    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    out_dir = tmp_path / "out"
    job = ConversionJob(
        id="conv-testtok2",
        source="gguf",
        model_type="generative",
        output_path=str(out_dir),
    )

    with (
        patch("mlx_server.conversion.gguf.mx.load") as mock_load,
        patch("mlx_server.conversion.gguf._remap_weights", return_value={}),
        patch("mlx_server.conversion.gguf._is_already_quantized", return_value=False),
        patch("mlx_server.conversion.gguf._build_config", return_value={"model_type": "mistral"}),
        patch("mlx_server.conversion.gguf._save_weights"),
        patch("mlx_server.conversion.gguf._download_tokenizer") as mock_download,
        patch("mlx_server.conversion.gguf._extract_tokenizer_from_gguf", return_value=True) as mock_extract,
    ):
        mock_load.return_value = ({}, {"general.architecture": "mistral"})
        convert_from_gguf(
            job=job,
            gguf_path=str(gguf_path),
            hf_tokenizer_repo="mistralai/Mistral-Nemo-Instruct-2407",
            prefer_hf_tokenizer=True,
            quantize=False,
        )

    mock_download.assert_called_once()
    mock_extract.assert_not_called()


def test_convert_from_gguf_falls_back_to_hf_when_extraction_fails(tmp_path: Path):
    """If extraction fails and hf_tokenizer_repo is provided, fallback to HF tokenizer."""
    from mlx_server.conversion.gguf import convert_from_gguf

    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    out_dir = tmp_path / "out"
    job = ConversionJob(
        id="conv-testtok3",
        source="gguf",
        model_type="generative",
        output_path=str(out_dir),
    )

    with (
        patch("mlx_server.conversion.gguf.mx.load") as mock_load,
        patch("mlx_server.conversion.gguf._remap_weights", return_value={}),
        patch("mlx_server.conversion.gguf._is_already_quantized", return_value=False),
        patch("mlx_server.conversion.gguf._build_config", return_value={"model_type": "mistral"}),
        patch("mlx_server.conversion.gguf._save_weights"),
        patch("mlx_server.conversion.gguf._download_tokenizer", return_value=["tokenizer.json"]) as mock_download,
        patch("mlx_server.conversion.gguf._extract_tokenizer_from_gguf", return_value=False) as mock_extract,
    ):
        mock_load.return_value = ({}, {"general.architecture": "mistral"})
        convert_from_gguf(
            job=job,
            gguf_path=str(gguf_path),
            hf_tokenizer_repo="mistralai/Mistral-Nemo-Instruct-2407",
            quantize=False,
        )

    mock_extract.assert_called_once()
    mock_download.assert_called_once()


def test_convert_from_gguf_raises_when_no_repo_and_extraction_fails(tmp_path: Path):
    """RuntimeError raised when GGUF extraction fails and no hf_tokenizer_repo provided."""
    from mlx_server.conversion.gguf import convert_from_gguf

    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    out_dir = tmp_path / "out"
    job = ConversionJob(
        id="conv-testtok4",
        source="gguf",
        model_type="generative",
        output_path=str(out_dir),
    )

    with (
        patch("mlx_server.conversion.gguf.mx.load") as mock_load,
        patch("mlx_server.conversion.gguf._remap_weights", return_value={}),
        patch("mlx_server.conversion.gguf._is_already_quantized", return_value=False),
        patch("mlx_server.conversion.gguf._build_config", return_value={"model_type": "mistral"}),
        patch("mlx_server.conversion.gguf._save_weights"),
        patch("mlx_server.conversion.gguf._download_tokenizer") as mock_download,
        patch("mlx_server.conversion.gguf._extract_tokenizer_from_gguf", return_value=False),
    ):
        mock_load.return_value = ({}, {"general.architecture": "mistral"})
        with pytest.raises(RuntimeError, match="hf_tokenizer_repo"):
            convert_from_gguf(
                job=job,
                gguf_path=str(gguf_path),
                quantize=False,
            )

    mock_download.assert_not_called()
