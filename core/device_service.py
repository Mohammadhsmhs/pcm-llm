"""
Device detection service following SOLID principles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DeviceInfo:
    """Information about the detected device."""

    device_type: str  # "cuda", "mps", "cpu"
    device_name: str
    torch_dtype: torch.dtype
    memory_gb: float
    supports_flash_attention: bool
    supports_quantization: bool


class IDeviceDetector(ABC):
    """Interface for device detection."""

    @abstractmethod
    def detect_device(self) -> DeviceInfo:
        """Detect the best available device for inference."""
        pass

    @abstractmethod
    def get_device_memory(self, device_type: str) -> float:
        """Get available memory for a device type in GB."""
        pass


class AppleSiliconDetector(IDeviceDetector):
    """Device detector optimized for Apple Silicon."""

    def detect_device(self) -> DeviceInfo:
        """Detect device for Apple Silicon systems."""
        if torch.cuda.is_available():
            return DeviceInfo(
                device_type="cuda",
                device_name="cuda",
                torch_dtype=torch.bfloat16,
                memory_gb=self.get_device_memory("cuda"),
                supports_flash_attention=True,
                supports_quantization=True,
            )
        elif torch.backends.mps.is_available():
            return DeviceInfo(
                device_type="mps",
                device_name="mps",
                torch_dtype=torch.float16,  # MPS works better with float16
                memory_gb=self.get_device_memory("mps"),
                supports_flash_attention=False,  # MPS has limited flash attention support
                supports_quantization=False,  # MPS doesn't support quantization
            )
        else:
            return DeviceInfo(
                device_type="cpu",
                device_name="cpu",
                torch_dtype=torch.float32,
                memory_gb=self.get_device_memory("cpu"),
                supports_flash_attention=False,
                supports_quantization=False,
            )

    def get_device_memory(self, device_type: str) -> float:
        """Get available memory for device type."""
        if device_type == "cuda":
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif device_type == "mps":
            # M2 typically has 8GB or 16GB unified memory
            return 8.0  # Conservative estimate
        else:
            return 4.0  # Conservative CPU estimate


class DeviceService:
    """Service for device detection and management."""

    def __init__(self, detector: Optional[IDeviceDetector] = None):
        self.detector = detector or AppleSiliconDetector()

    def get_optimal_device(self) -> DeviceInfo:
        """Get the optimal device for inference."""
        return self.detector.detect_device()

    def validate_model_compatibility(
        self, device_info: DeviceInfo, model_name: str
    ) -> DeviceInfo:
        """Validate and adjust device config based on model compatibility."""
        # Check for quantized model names that might not work on MPS/CPU
        model_lower = model_name.lower()
        incompatible_patterns = ["-bnb-", "bnb-", "-4bit", "-8bit", "gguf"]

        if device_info.device_type in ("mps", "cpu"):
            if any(pattern in model_lower for pattern in incompatible_patterns):
                print(
                    f"⚠️  Model '{model_name}' appears to be quantized and may not work on {device_info.device_type}"
                )
                print(
                    "   Consider using a non-quantized model for better compatibility"
                )
                return DeviceInfo(
                    device_type="cpu",  # Fallback to CPU for incompatible models
                    device_name="cpu",
                    torch_dtype=torch.float32,
                    memory_gb=self.detector.get_device_memory("cpu"),
                    supports_flash_attention=False,
                    supports_quantization=False,
                )

        return device_info


# Global device service instance
device_service = DeviceService()
