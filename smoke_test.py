"""
Smoke test for the MLX server.

Requires the server to be running:
    uv run mlx-server

Run this script with:
    uv run python smoke_test.py [--test inference|hf|gguf|all]

Tests:
  inference — register a pre-built MLX model and run a chat completion
  hf        — convert SmolLM2-135M from HF then run inference
  gguf      — convert a local GGUF (DeepSeek-R1-1.5B) then run inference
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

BASE = "http://localhost:1234"

# ---------------------------------------------------------------------------
# Tiny models used in each test
# ---------------------------------------------------------------------------

# Tiny public HF model — mlx_lm loads HF format directly (no pre-conversion needed)
INFERENCE_HF_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Non-MLX HF model to run through convert_from_hf
HF_CONVERT_REPO   = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Local GGUF on NAS — update path if needed
GGUF_PATH = "/Volumes/jvnas/LLM_models/lmstudio-community/Ministral-3-14B-Reasoning-2512-GGUF/Ministral-3-14B-Reasoning-2512-Q8_0.gguf"
GGUF_TOKENIZER = "mistralai/Mistral-Nemo-Instruct-2407"  # Ministral-3 uses the Nemo tokenizer

# Output dirs — default to MLX_MODEL_DIR (set in env or server config)
def _model_dir() -> Path:
    """Respects MLX_MODEL_DIR env var, same as the server."""
    import os
    d = os.environ.get("MLX_MODEL_DIR")
    return Path(d) if d else Path.home() / ".cache" / "mlx-server" / "models"

HF_CONVERT_OUT   = str(_model_dir() / "test-smollm2-135m")
GGUF_CONVERT_OUT = str(_model_dir() / "ministral-14b-mlx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(resp: requests.Response, label: str) -> dict:
    if not resp.ok:
        print(f"  FAIL [{resp.status_code}] {label}")
        try:
            print(f"       {resp.json()}")
        except Exception:
            print(f"       {resp.text[:200]}")
        sys.exit(1)
    data = resp.json()
    print(f"  OK   {label}")
    return data


def _poll_interval(size_gb: float) -> int:
    """Poll interval based on model size — small models check often, large ones wait."""
    if size_gb < 1:
        return 5
    if size_gb < 5:
        return 20
    if size_gb < 15:
        return 120   # ~2 min — shard writes to NAS take several minutes
    return 300       # 5 min — very large models


def _wait_for_job(job_id: str, size_gb: float = 0.1, timeout: int = 7200) -> dict:
    """Poll until job is completed or failed."""
    interval = _poll_interval(size_gb)
    start = time.time()
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    last_progress = ""
    spin = 0
    while True:
        elapsed = int(time.time() - start)
        try:
            resp = requests.get(f"{BASE}/v1/convert/{job_id}", timeout=30)
            job = resp.json()
        except (requests.ConnectionError, requests.ReadTimeout):
            print(f"\n  ! server slow/unreachable at {elapsed}s — retrying...")
            time.sleep(interval)
            continue
        status = job["status"]
        progress = job.get("progress", "")

        if progress != last_progress:
            last_progress = progress
            print(f"  [{elapsed:>4}s] {progress}")

        print(f"  {SPINNER[spin % len(SPINNER)]} {elapsed}s elapsed...", end="\r", flush=True)
        spin += 1

        if status == "completed":
            print()
            return job
        if status == "failed":
            print()
            print(f"  FAIL job error: {job.get('error')}")
            sys.exit(1)
        if elapsed > timeout:
            print()
            print(f"  FAIL job timed out after {timeout}s")
            sys.exit(1)
        time.sleep(interval)


def _chat(model_id: str, prompt: str = "Say 'hello' in exactly one word.") -> str:
    resp = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0.0,
    })
    _check(resp, f"POST /v1/chat/completions model={model_id}")
    text = resp.json()["choices"][0]["message"]["content"]
    print(f"       response: {text!r}")
    return text


def _download_mlx_model(hf_repo: str) -> str:
    """Download a model via huggingface_hub and return local path."""
    from huggingface_hub import snapshot_download
    print(f"  Downloading {hf_repo} ...")
    # token=False → explicit anonymous access (avoids 401 from stale env tokens)
    path = snapshot_download(repo_id=hf_repo, token=False)
    print(f"  Downloaded to {path}")
    return path


# ---------------------------------------------------------------------------
# Test: direct inference with a pre-built MLX model
# ---------------------------------------------------------------------------

def test_inference():
    print("\n=== TEST: inference ===")

    model_id = "smollm2-135m-test"

    # 1. Check if already registered (idempotent)
    resp = requests.get(f"{BASE}/v1/models/{model_id}")
    if resp.status_code == 200:
        print(f"  {model_id} already registered, skipping download")
        local_path = resp.json()["path"]
    else:
        local_path = _download_mlx_model(INFERENCE_HF_REPO)
        _check(
            requests.post(f"{BASE}/v1/models/register", json={
                "id": model_id,
                "path": local_path,
                "type": "generative",
            }),
            f"POST /v1/models/register id={model_id}",
        )

    # 2. Chat
    _chat(model_id)
    print("  PASS inference")


# ---------------------------------------------------------------------------
# Test: HF conversion
# ---------------------------------------------------------------------------

def test_hf_conversion():
    print("\n=== TEST: hf conversion ===")

    model_id = "smollm2-135m-converted"
    out = Path(HF_CONVERT_OUT)

    if out.exists():
        print(f"  Output already exists: {out}")
    else:
        resp = _check(
            requests.post(f"{BASE}/v1/convert/hf", json={
                "hf_repo": HF_CONVERT_REPO,
                "output_path": str(out),
                "model_type": "generative",
                "quantize": True,
                "q_bits": 4,
                "register_as": model_id,
            }),
            "POST /v1/convert/hf",
        )
        print(f"  Job: {resp['job_id']}")
        _wait_for_job(resp["job_id"])  # HF: small model, default size

    # Register if not already (job may have done it)
    resp = requests.get(f"{BASE}/v1/models/{model_id}")
    if resp.status_code != 200:
        _check(
            requests.post(f"{BASE}/v1/models/register", json={
                "id": model_id,
                "path": str(out),
                "type": "generative",
            }),
            f"POST /v1/models/register id={model_id}",
        )

    _chat(model_id)
    print("  PASS hf conversion")


# ---------------------------------------------------------------------------
# Test: GGUF conversion
# ---------------------------------------------------------------------------

def test_gguf_conversion():
    print("\n=== TEST: gguf conversion ===")

    if not Path(GGUF_PATH).exists():
        print(f"  SKIP — GGUF not found at {GGUF_PATH}")
        print("         Update GGUF_PATH at the top of this script.")
        return

    model_id = "deepseek-r1-1.5b-mlx"
    out = Path(GGUF_CONVERT_OUT)

    if out.exists():
        print(f"  Output already exists: {out}")
    else:
        resp = _check(
            requests.post(f"{BASE}/v1/convert/gguf", json={
                "gguf_path": GGUF_PATH,
                "hf_tokenizer_repo": GGUF_TOKENIZER,
                "output_path": str(out),
                "quantize": False,   # Q8_0 source — already quantized, skip re-quant
                "register_as": model_id,
            }),
            "POST /v1/convert/gguf",
        )
        gguf_size_gb = Path(GGUF_PATH).stat().st_size / 1024 ** 3
        print(f"  Job: {resp['job_id']}")
        _wait_for_job(resp["job_id"], size_gb=gguf_size_gb)

    # Register if needed
    resp = requests.get(f"{BASE}/v1/models/{model_id}")
    if resp.status_code != 200:
        _check(
            requests.post(f"{BASE}/v1/models/register", json={
                "id": model_id,
                "path": str(out),
                "type": "generative",
            }),
            f"POST /v1/models/register id={model_id}",
        )

    _chat(model_id)
    print("  PASS gguf conversion")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        choices=["inference", "hf", "gguf", "all"],
        default="all",
    )
    args = parser.parse_args()

    # Confirm server is up
    try:
        requests.get(BASE + "/docs", timeout=3)
    except requests.ConnectionError:
        print(f"Server not reachable at {BASE}")
        print("Start it with: uv run mlx-server")
        sys.exit(1)

    if args.test in ("inference", "all"):
        test_inference()
    if args.test in ("hf", "all"):
        test_hf_conversion()
    if args.test in ("gguf", "all"):
        test_gguf_conversion()

    print("\nAll selected tests passed.")


if __name__ == "__main__":
    main()
