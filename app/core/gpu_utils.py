"""
GPU & Docker utilities for dots.ocr lifecycle management.

Responsibilities:
  • Detect GPU hardware (VRAM, compute capability, driver)
  • Calculate safe vLLM parameters (gpu-memory-utilization, max-model-len)
  • Auto-pull, start, health-check, and stop dots.ocr Docker container
  • Graceful degradation: GPU → CPU → skip (falls through to Tesseract)

Usage:
  from app.core.gpu_utils import GPUInfo, DotsOCRManager

All ENV vars expected under PDF_DOTS_OCR_* prefix via PDFSettings.
"""

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("ai-ca")

# ═════════════════════════════════════════════════════════════════
# GPU DETECTION
# ═════════════════════════════════════════════════════════════════

# VRAM overhead reserved by CUDA driver + runtime (empirical, in GiB)
_CUDA_OVERHEAD_GIB = 0.8
# Model footprint estimates (used by safe_max_model_len)
_MODEL_FOOTPRINT_GIB = 3.5   # bf16 full precision (for GPUs ≥ 8 GB)
_AWQ_FOOTPRINT_GIB = 1.8     # AWQ 4-bit quantized (for GPUs < 8 GB)
# Minimum usable VRAM for VLM OCR (AWQ + small KV cache)
_MIN_VRAM_GIB = 4.0


@dataclass
class GPUInfo:
    """Detected GPU hardware information."""
    available: bool = False
    device_name: str = ""
    vram_total_gib: float = 0.0
    vram_free_gib: float = 0.0
    cuda_version: str = ""
    compute_capability: str = ""
    driver_version: str = ""

    @property
    def can_run_dots_ocr(self) -> bool:
        """Whether this GPU has enough VRAM for dots.ocr."""
        return self.available and self.vram_total_gib >= _MIN_VRAM_GIB

    @property
    def safe_memory_utilization(self) -> float:
        """
        Calculate safe gpu-memory-utilization for vLLM.

        Note: GPUs often report slightly under nominal (e.g. 5.99 GiB for "6 GB"),
        so we use -0.1 GiB tolerance on each threshold.

        Strategy:
          ≥ 24 GB (A100, RTX 4090)  → 0.90  (plenty of headroom)
          ≥ 12 GB (RTX 3080, 4070)  → 0.80
          ≥  8 GB (RTX 3060Ti)      → 0.70
          ≥  6 GB (RTX 3060 Laptop) → 0.60  (tight — model only, small KV)
          <  6 GB                   → 0.50  (best-effort)
        """
        vram = self.vram_total_gib
        if vram >= 23.9:
            return 0.90
        elif vram >= 11.9:
            return 0.80
        elif vram >= 7.9:
            return 0.70
        elif vram >= 5.9:
            return 0.60
        else:
            return 0.50

    @property
    def safe_max_model_len(self) -> int:
        """
        Calculate safe max-model-len based on available VRAM after model loading.

        Uses AWQ footprint (1.8 GiB) for GPUs < 8 GB, bf16 (3.5 GiB) otherwise.

        Remaining VRAM after model → KV cache slots:
          ≥ 4 GB remaining → 2048 (full context)
          ≥ 2 GB remaining → 1024
          < 2 GB           →  512 (minimal context, still functional)
        """
        # Pick footprint matching the model variant we'll auto-select
        footprint = _MODEL_FOOTPRINT_GIB if self.vram_total_gib >= 7.9 else _AWQ_FOOTPRINT_GIB
        usable = self.vram_total_gib * self.safe_memory_utilization
        remaining = usable - footprint
        if remaining >= 4.0:
            return 2048
        elif remaining >= 2.0:
            return 1024
        else:
            return 512


def detect_gpu() -> GPUInfo:
    """
    Detect GPU hardware via PyTorch (preferred) or nvidia-smi (fallback).

    Returns GPUInfo with available=False if no GPU is found.
    """
    info = GPUInfo()

    # ── Strategy 1: PyTorch CUDA (most reliable) ──
    try:
        import torch
        if torch.cuda.is_available():
            info.available = True
            info.device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.vram_total_gib = props.total_memory / (1024 ** 3)
            info.cuda_version = torch.version.cuda or ""
            info.compute_capability = f"{props.major}.{props.minor}"

            # Free memory (may differ from total if other processes use GPU)
            free_mem, _ = torch.cuda.mem_get_info(0)
            info.vram_free_gib = free_mem / (1024 ** 3)

            logger.info(
                f"GPU detected via PyTorch: {info.device_name}, "
                f"VRAM: {info.vram_total_gib:.1f} GiB "
                f"(free: {info.vram_free_gib:.1f} GiB), "
                f"CUDA: {info.cuda_version}, "
                f"Compute: {info.compute_capability}"
            )
            return info
    except Exception as e:
        logger.debug(f"PyTorch GPU detection failed: {e}")

    # ── Strategy 2: nvidia-smi CLI (Docker-friendly, no torch needed) ──
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 4:
                    info.available = True
                    info.device_name = parts[0]
                    info.vram_total_gib = float(parts[1]) / 1024  # MiB → GiB
                    info.vram_free_gib = float(parts[2]) / 1024
                    info.driver_version = parts[3]
                    logger.info(
                        f"GPU detected via nvidia-smi: {info.device_name}, "
                        f"VRAM: {info.vram_total_gib:.1f} GiB "
                        f"(free: {info.vram_free_gib:.1f} GiB), "
                        f"Driver: {info.driver_version}"
                    )
                    return info
        except Exception as e:
            logger.debug(f"nvidia-smi GPU detection failed: {e}")

    logger.info("No CUDA GPU detected — dots.ocr will be skipped")
    return info


# ═════════════════════════════════════════════════════════════════
# SELF-HOSTED VLM DOCKER LIFECYCLE MANAGER
# ═════════════════════════════════════════════════════════════════

# Model selection matrix (all ungated, free — no HF token needed):
#   ≥ 8 GB VRAM → bf16 full precision (best quality)
#   4–8 GB VRAM → AWQ 4-bit quantized (fits 6 GB!)
#   < 4 GB VRAM → skip (fall through to Tesseract)
_MODEL_VARIANTS = {
    "full": "Qwen/Qwen2.5-VL-3B-Instruct",        # bf16, ~7.5 GiB VRAM
    "awq":  "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",    # 4-bit, ~1.8 GiB weights
}
_AWQ_MODEL_FOOTPRINT_GIB = 1.8   # AWQ weights in VRAM
_FULL_MODEL_FOOTPRINT_GIB = 3.5  # bf16 weights in VRAM

_DEFAULT_DOCKER_IMAGE = "vllm/vllm-openai:latest"
_DEFAULT_MODEL_NAME = _MODEL_VARIANTS["awq"]  # safe default for any GPU ≥ 4 GB
_CONTAINER_NAME = "ai-ca-dots-ocr"
_HEALTH_CHECK_RETRIES = 20
_HEALTH_CHECK_INTERVAL_S = 6
_STARTUP_GRACE_S = 30  # WSL + torch.compile needs extra bootstrap time


def _select_model_for_vram(vram_gib: float, user_model: Optional[str] = None) -> str:
    """
    Auto-select the best model variant for available VRAM.

    If user explicitly set PDF_DOTS_OCR_MODEL, respect their choice.
    Otherwise, pick the best variant that fits the GPU.
    """
    if user_model and user_model not in _MODEL_VARIANTS.values():
        # User specified a custom model (e.g. dots.ocr) — respect it
        return user_model

    if vram_gib >= 7.9:
        chosen = _MODEL_VARIANTS["full"]
        logger.info(f"VRAM {vram_gib:.1f} GiB ≥ 8: using bf16 model ({chosen})")
    else:
        chosen = _MODEL_VARIANTS["awq"]
        logger.info(f"VRAM {vram_gib:.1f} GiB < 8: using AWQ 4-bit model ({chosen})")
    return chosen


@dataclass
class DotsOCRManager:
    """
    Manages the self-hosted VLM Docker container lifecycle.

    Flow:
      1. detect_gpu() → hardware check
      2. _select_model_for_vram() → pick bf16 vs AWQ based on VRAM
      3. ensure_running() → pull image if needed, start container with safe params
      4. health_check() → wait for /v1/models endpoint
      5. stop() → clean shutdown
    """
    container_port: int = 8000
    host_port: int = 5555
    docker_image: str = _DEFAULT_DOCKER_IMAGE
    model_name: str = _DEFAULT_MODEL_NAME
    hf_token: Optional[str] = None
    gpu_info: GPUInfo = field(default_factory=GPUInfo)
    _container_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        self.gpu_info = detect_gpu()
        # Auto-detect HF token from environment if not passed
        if not self.hf_token:
            self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        # Auto-select best model variant for detected GPU
        self.model_name = _select_model_for_vram(
            self.gpu_info.vram_total_gib, self.model_name
        )

    @property
    def endpoint(self) -> str:
        """The HTTP endpoint for dots.ocr API calls."""
        return f"http://localhost:{self.host_port}"

    @property
    def docker_available(self) -> bool:
        """Check if Docker CLI is available."""
        return shutil.which("docker") is not None

    def ensure_running(self) -> bool:
        """
        Ensure dots.ocr container is running with safe GPU parameters.

        Returns True if container is running and healthy, False otherwise.
        Falls back gracefully:
          1. No Docker → return False (fall through to Tesseract)
          2. No GPU    → return False (model needs CUDA)
          3. OOM-safe  → auto-tune gpu-memory-utilization based on VRAM
        """
        if not self.docker_available:
            logger.warning("Docker not available — dots.ocr container cannot be started")
            return False

        if not self.gpu_info.can_run_dots_ocr:
            logger.warning(
                f"Insufficient GPU for dots.ocr: "
                f"{self.gpu_info.device_name or 'none'} "
                f"({self.gpu_info.vram_total_gib:.1f} GiB < {_MIN_VRAM_GIB} GiB minimum)"
            )
            return False

        # Check if already running
        if self._is_container_running():
            logger.info(f"dots.ocr container '{_CONTAINER_NAME}' already running")
            return self._wait_for_healthy()

        # Remove stale container if exists
        self._remove_container()

        # Pull image if not present
        if not self._image_exists():
            logger.info(f"Pulling dots.ocr Docker image: {self.docker_image}...")
            try:
                subprocess.run(
                    ["docker", "pull", self.docker_image],
                    capture_output=True, text=True, timeout=600,
                    check=True,
                )
                logger.info("dots.ocr image pulled successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to pull dots.ocr image: {e.stderr}")
                return False
            except subprocess.TimeoutExpired:
                logger.error("dots.ocr image pull timed out (10 min)")
                return False

        # Build docker run command with safe GPU parameters
        mem_util = self.gpu_info.safe_memory_utilization
        max_model_len = self.gpu_info.safe_max_model_len

        cmd = [
            "docker", "run", "-d",
            "--name", _CONTAINER_NAME,
            "--gpus", "all",
            "-p", f"{self.host_port}:{self.container_port}",
            "--restart", "unless-stopped",
        ]

        # Pass HF token for gated model access
        if self.hf_token:
            cmd.extend(["-e", f"HF_TOKEN={self.hf_token}"])

        cmd.extend([
            self.docker_image,
            "--model", self.model_name,
            "--served-model-name", "model",
            "--trust-remote-code",
            "--gpu-memory-utilization", str(mem_util),
            "--max-model-len", str(max_model_len),
        ])

        logger.info(
            f"Starting dots.ocr container: "
            f"GPU={self.gpu_info.device_name}, "
            f"VRAM={self.gpu_info.vram_total_gib:.1f}GiB, "
            f"mem_util={mem_util}, "
            f"max_model_len={max_model_len}, "
            f"port={self.host_port}"
        )

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=True,
            )
            self._container_id = result.stdout.strip()[:12]
            logger.info(f"dots.ocr container started: {self._container_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start dots.ocr container: {e.stderr}")
            return False

        # Wait for model to load and become healthy
        return self._wait_for_healthy()

    def stop(self):
        """Stop and remove the dots.ocr container."""
        if self.docker_available:
            self._remove_container()
            logger.info("dots.ocr container stopped")

    def is_healthy(self) -> bool:
        """Quick health check — is the endpoint responding?"""
        try:
            import httpx
            resp = httpx.get(
                f"{self.endpoint}/v1/models",
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ── Private helpers ──────────────────────────────────────────

    def _is_container_running(self) -> bool:
        """Check if our named container is currently running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", _CONTAINER_NAME],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0 and "true" in result.stdout.lower()
        except Exception:
            return False

    def _remove_container(self):
        """Force-remove container (ignores errors if not present)."""
        try:
            subprocess.run(
                ["docker", "rm", "-f", _CONTAINER_NAME],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            pass

    def _image_exists(self) -> bool:
        """Check if the Docker image is already pulled."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.docker_image],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _wait_for_healthy(self) -> bool:
        """
        Wait for dots.ocr to become healthy (model loading takes time).

        Checks /v1/models endpoint every _HEALTH_CHECK_INTERVAL_S seconds,
        up to _HEALTH_CHECK_RETRIES times.
        """
        logger.info(
            f"Waiting for dots.ocr to load model "
            f"(up to {_HEALTH_CHECK_RETRIES * _HEALTH_CHECK_INTERVAL_S}s)..."
        )

        # Initial grace period for container bootstrap
        time.sleep(_STARTUP_GRACE_S)

        for attempt in range(1, _HEALTH_CHECK_RETRIES + 1):
            # First check container is still running (didn't crash)
            if not self._is_container_running():
                logs = self._get_container_logs(tail=15)
                logger.error(
                    f"dots.ocr container exited during startup. "
                    f"Likely OOM or config error.\nLast logs:\n{logs}"
                )
                return False

            if self.is_healthy():
                logger.info(f"dots.ocr healthy after {attempt} checks")
                return True

            logger.debug(
                f"dots.ocr not ready (attempt {attempt}/{_HEALTH_CHECK_RETRIES})"
            )
            time.sleep(_HEALTH_CHECK_INTERVAL_S)

        # Final attempt — get logs for debugging
        logs = self._get_container_logs(tail=10)
        logger.warning(
            f"dots.ocr did not become healthy after "
            f"{_HEALTH_CHECK_RETRIES * _HEALTH_CHECK_INTERVAL_S}s. "
            f"Falling back to local OCR.\nLast logs:\n{logs}"
        )
        return False

    def _get_container_logs(self, tail: int = 20) -> str:
        """Get recent container logs for debugging."""
        try:
            result = subprocess.run(
                ["docker", "logs", _CONTAINER_NAME, "--tail", str(tail)],
                capture_output=True, text=True, timeout=5,
            )
            return (result.stdout + result.stderr).strip()
        except Exception:
            return "<unable to retrieve logs>"
