"""
Startup import-cost guards.

Importing the FastAPI app must not pull in the heavy inference backends.
mlx_lm imports transformers, which imports torch — several seconds of cold
startup before the server can answer even /v1/models. The managers already
defer mlx_embeddings and mlx_vlm; mlx_lm should behave the same way.

These run in a subprocess: the test session itself has the backends loaded, so
sys.modules in-process tells us nothing.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _modules_after_importing(target: str) -> set[str]:
    """Return the top-level modules present after importing ``target`` fresh."""
    code = (
        "import sys, json\n"
        f"import {target}\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"import of {target} failed:\n{proc.stderr}"
    import json

    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("backend", ["torch", "transformers", "mlx_lm"])
def test_importing_app_does_not_load_inference_backend(backend: str):
    loaded = _modules_after_importing("mlx_server.api.app")
    assert backend not in loaded, (
        f"importing mlx_server.api.app pulled in {backend!r}; "
        "the inference backends must stay lazy so startup is not blocked"
    )


def test_generative_manager_still_exposes_its_public_api():
    """The lazy import must not change what the module offers."""
    loaded = _modules_after_importing("mlx_server.managers.generative")
    assert "mlx_lm" not in loaded
