"""
HuggingFace → MLX conversion.

Downloads safetensors from HF (or uses local HF cache) and converts
to MLX format using the appropriate backend:
  generative → mlx_lm.convert
  embedding  → mlx_embeddings.convert
  vlm        → mlx_vlm.convert
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .jobs import ConversionJob

logger = logging.getLogger(__name__)


def convert_from_hf(
    job: ConversionJob,
    hf_repo: str,
    quantize: bool = True,
    q_bits: int = 4,
    q_group_size: int = 64,
    dtype: Optional[str] = None,
) -> None:
    output = Path(job.output_path)
    if output.exists():
        raise FileExistsError(
            f"Output path already exists: {output}. "
            "Delete it or choose a different path."
        )

    job.progress = f"Downloading / converting from {hf_repo}"
    logger.info(job.progress)

    if job.model_type == "generative":
        from mlx_lm.convert import convert

        convert(
            hf_path=hf_repo,
            mlx_path=str(output),
            quantize=quantize,
            q_bits=q_bits,
            q_group_size=q_group_size,
            dtype=dtype,
        )

    elif job.model_type == "embedding":
        from mlx_embeddings.convert import convert

        convert(
            hf_path=hf_repo,
            mlx_path=str(output),
            quantize=quantize,
            q_bits=q_bits,
        )

    elif job.model_type == "vlm":
        from mlx_vlm.convert import convert

        convert(
            hf_path=hf_repo,
            mlx_path=str(output),
            quantize=quantize,
            q_bits=q_bits,
            dtype=dtype or "bfloat16",
        )

    else:
        raise ValueError(f"Unknown model_type: {job.model_type}")

    job.progress = f"Saved to {output}"
    logger.info(f"Conversion complete: {output}")
