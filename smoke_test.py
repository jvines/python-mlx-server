"""
Smoke test for the MLX server.

Requires the server to be running:
    uv run mlx-server

Run this script with:
    uv run python smoke_test.py [options]

Tests:
  inference — register a pre-built MLX model (downloaded from HF) and run chat
  hf        — convert SmolLM2-135M from HF to MLX, then run inference
  gguf      — convert a local GGUF file to MLX, then run inference
              Requires --gguf-path and --gguf-tokenizer

Examples:
    uv run python smoke_test.py --test inference
    uv run python smoke_test.py --test hf
    uv run python smoke_test.py --test gguf \\
        --gguf-path /path/to/model.gguf \\
        --gguf-tokenizer meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tiny public HF model — mlx_lm loads HF format directly (no pre-conversion needed)
INFERENCE_HF_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Tiny non-MLX HF model to run through convert_from_hf
HF_CONVERT_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _model_dir() -> Path:
    """Output directory for converted models. Respects MLX_MODEL_DIR env var."""
    d = os.environ.get("MLX_MODEL_DIR")
    return Path(d) if d else Path.home() / ".cache" / "mlx-server" / "models"


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
    """Poll interval based on source model size."""
    if size_gb < 1:
        return 5
    if size_gb < 5:
        return 20
    if size_gb < 15:
        return 120
    return 300


def _wait_for_job(job_id: str, base: str, size_gb: float = 0.1, timeout: int = 7200) -> dict:
    """Poll until job is completed or failed."""
    interval = _poll_interval(size_gb)
    start = time.time()
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    last_progress = ""
    spin = 0
    while True:
        elapsed = int(time.time() - start)
        try:
            resp = requests.get(f"{base}/v1/convert/{job_id}", timeout=30)
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


def _chat(model_id: str, base: str, prompt: str = "Say 'hello' in exactly one word.") -> str:
    resp = requests.post(f"{base}/v1/chat/completions", json={
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
    path = snapshot_download(repo_id=hf_repo, token=False)
    print(f"  Downloaded to {path}")
    return path


# ---------------------------------------------------------------------------
# Test: direct inference with a pre-built MLX model
# ---------------------------------------------------------------------------


def test_inference(base: str) -> None:
    print("\n=== TEST: inference ===")

    model_id = "smoke-inference"

    resp = requests.get(f"{base}/v1/models/{model_id}")
    if resp.status_code == 200:
        print(f"  {model_id} already registered, skipping download")
    else:
        local_path = _download_mlx_model(INFERENCE_HF_REPO)
        _check(
            requests.post(f"{base}/v1/models/register", json={
                "id": model_id,
                "path": local_path,
                "type": "generative",
            }),
            f"POST /v1/models/register id={model_id}",
        )

    _chat(model_id, base)
    print("  PASS inference")


# ---------------------------------------------------------------------------
# Test: HF conversion
# ---------------------------------------------------------------------------


def test_hf_conversion(base: str) -> None:
    print("\n=== TEST: hf conversion ===")

    model_id = "smoke-hf-converted"
    out = _model_dir() / "smoke-hf-converted"

    if out.exists():
        print(f"  Output already exists: {out}")
    else:
        resp = _check(
            requests.post(f"{base}/v1/convert/hf", json={
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
        _wait_for_job(resp["job_id"], base)

    resp = requests.get(f"{base}/v1/models/{model_id}")
    if resp.status_code != 200:
        _check(
            requests.post(f"{base}/v1/models/register", json={
                "id": model_id,
                "path": str(out),
                "type": "generative",
            }),
            f"POST /v1/models/register id={model_id}",
        )

    _chat(model_id, base)
    print("  PASS hf conversion")


# ---------------------------------------------------------------------------
# Test: GGUF conversion
# ---------------------------------------------------------------------------


def test_gguf_conversion(base: str, gguf_path: str, gguf_tokenizer: str) -> None:
    print("\n=== TEST: gguf conversion ===")

    gguf = Path(gguf_path)
    if not gguf.exists():
        print(f"  FAIL — GGUF not found: {gguf_path}")
        sys.exit(1)

    # Derive a stable model ID and output dir from the GGUF filename
    model_id = f"smoke-gguf-{gguf.stem[:32].lower().replace('_', '-')}"
    out = _model_dir() / model_id

    if out.exists():
        print(f"  Output already exists: {out}")
    else:
        gguf_size_gb = gguf.stat().st_size / 1024 ** 3
        resp = _check(
            requests.post(f"{base}/v1/convert/gguf", json={
                "gguf_path": str(gguf),
                "hf_tokenizer_repo": gguf_tokenizer,
                "output_path": str(out),
                "quantize": False,
                "register_as": model_id,
            }),
            "POST /v1/convert/gguf",
        )
        print(f"  Job: {resp['job_id']}  ({gguf_size_gb:.1f} GB, polling every {_poll_interval(gguf_size_gb)}s)")
        _wait_for_job(resp["job_id"], base, size_gb=gguf_size_gb)

    resp = requests.get(f"{base}/v1/models/{model_id}")
    if resp.status_code != 200:
        _check(
            requests.post(f"{base}/v1/models/register", json={
                "id": model_id,
                "path": str(out),
                "type": "generative",
            }),
            f"POST /v1/models/register id={model_id}",
        )

    _chat(model_id, base)
    print("  PASS gguf conversion")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the MLX server.")
    parser.add_argument("--base-url", default="http://localhost:8080",
                        help="Server base URL (default: http://localhost:8080)")
    parser.add_argument("--test", choices=["inference", "hf", "gguf", "all"], default="all")
    parser.add_argument("--gguf-path", help="Path to a local GGUF file (required for --test gguf)")
    parser.add_argument("--gguf-tokenizer", help="HF repo ID for the GGUF tokenizer (required for --test gguf)")
    args = parser.parse_args()

    if args.test in ("gguf", "all") and args.test == "gguf":
        if not args.gguf_path or not args.gguf_tokenizer:
            parser.error("--test gguf requires --gguf-path and --gguf-tokenizer")

    base = args.base_url.rstrip("/")

    try:
        requests.get(base + "/docs", timeout=3)
    except requests.ConnectionError:
        print(f"Server not reachable at {base}")
        print("Start it with: uv run mlx-server")
        sys.exit(1)

    if args.test in ("inference", "all"):
        test_inference(base)
    if args.test in ("hf", "all"):
        test_hf_conversion(base)
    if args.test in ("gguf",):
        test_gguf_conversion(base, args.gguf_path, args.gguf_tokenizer)

    print("\nAll selected tests passed.")


if __name__ == "__main__":
    main()