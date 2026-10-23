"""Environment / GPU check for DeepDefender.

Run directly (`python -m src.environment`) or via `python -m src.train --check`.
Must never crash just because torch/transformers aren't installed yet -- Phase 1
(dataset + preprocessing) has to work without them, so this reports what's
missing instead of raising.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class EnvironmentReport:
    python_version: str
    platform: str
    torch_installed: bool
    torch_version: Optional[str]
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    gpu_total_vram_mb: Optional[int]
    transformers_installed: bool
    transformers_version: Optional[str]
    nvidia_smi_found: bool


def _nvidia_smi_summary() -> Optional[str]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def check_environment() -> EnvironmentReport:
    torch_installed = False
    torch_version = None
    cuda_available = False
    cuda_version = None
    gpu_name = None
    gpu_total_vram_mb = None

    try:
        import torch  # type: ignore

        torch_installed = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            gpu_total_vram_mb = int(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            )
    except ImportError:
        pass

    if not cuda_available:
        smi = _nvidia_smi_summary()
        if smi and "," in smi:
            name, mem = smi.split(",", 1)
            gpu_name = gpu_name or name.strip()
            try:
                gpu_total_vram_mb = gpu_total_vram_mb or int(
                    float(mem.strip().split()[0])
                )
            except (ValueError, IndexError):
                pass

    transformers_installed = False
    transformers_version = None
    try:
        import transformers  # type: ignore

        transformers_installed = True
        transformers_version = transformers.__version__
    except ImportError:
        pass

    return EnvironmentReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_installed=torch_installed,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_total_vram_mb=gpu_total_vram_mb,
        transformers_installed=transformers_installed,
        transformers_version=transformers_version,
        nvidia_smi_found=shutil.which("nvidia-smi") is not None,
    )


def print_report(report: EnvironmentReport) -> None:
    print("=" * 60)
    print("DeepDefender - Environment Check")
    print("=" * 60)
    print(f"Python version        : {report.python_version}")
    print(f"Platform              : {report.platform}")
    print(f"nvidia-smi found      : {report.nvidia_smi_found}")
    print(f"GPU name              : {report.gpu_name or 'not detected'}")
    if report.gpu_total_vram_mb:
        print(f"GPU total VRAM        : {report.gpu_total_vram_mb} MB "
              f"(~{report.gpu_total_vram_mb / 1024:.1f} GB)")
    else:
        print("GPU total VRAM        : not detected")
    print(f"torch installed       : {report.torch_installed} "
          f"({report.torch_version or 'n/a'})")
    print(f"CUDA available (torch): {report.cuda_available} "
          f"(cuda {report.cuda_version or 'n/a'})")
    print(f"transformers installed: {report.transformers_installed} "
          f"({report.transformers_version or 'n/a'})")
    print("=" * 60)

    if not report.torch_installed:
        print("torch is NOT installed. Install it before training. Example "
              "(check https://pytorch.org for the exact command for your "
              "CUDA driver version):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    elif not report.cuda_available:
        print("torch is installed but CUDA is NOT available to it. Training "
              "will fall back to CPU (very slow for transformers). If you "
              "have an NVIDIA GPU, reinstall torch with a CUDA-enabled build.")
    elif report.gpu_total_vram_mb and report.gpu_total_vram_mb < 6000:
        print(f"GPU VRAM is limited (~{report.gpu_total_vram_mb} MB). The "
              "default training config in configs/config.yaml is already "
              "conservative (batch_size=4, grad_accum=4, max_length=128, "
              "fp16=true) for this reason. If you hit CUDA out-of-memory, "
              "lower batch_size first, then max_length, then enable "
              "gradient_checkpointing.")

    if not report.transformers_installed:
        print("transformers is NOT installed. Install project requirements "
              "with: pip install -r requirements.txt")


if __name__ == "__main__":
    print_report(check_environment())
