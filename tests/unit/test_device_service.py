"""
Unit tests for device service.
"""

import unittest
from unittest.mock import Mock, patch
from tests import UnitTestCase
from core.device_service import (
    DeviceService,
    AppleSiliconDetector,
    DeviceInfo
)


class TestDeviceInfo(UnitTestCase):
    """Test DeviceInfo data class."""

    def test_device_info_creation(self):
        """Test creating a DeviceInfo instance."""
        info = DeviceInfo(
            device_type="mps",
            device_name="mps",
            torch_dtype="float16",
            memory_gb=8.0,
            supports_flash_attention=False,
            supports_quantization=False
        )

        self.assertEqual(info.device_type, "mps")
        self.assertEqual(info.device_name, "mps")
        self.assertEqual(info.torch_dtype, "float16")
        self.assertEqual(info.memory_gb, 8.0)
        self.assertFalse(info.supports_flash_attention)
        self.assertFalse(info.supports_quantization)


class TestAppleSiliconDetector(UnitTestCase):
    """Test AppleSiliconDetector implementation."""

    def setUp(self):
        super().setUp()
        self.detector = AppleSiliconDetector()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_detect_device_cuda(self, mock_props, mock_cuda_available):
        """Test device detection when CUDA is available."""
        mock_device_props = Mock()
        mock_device_props.total_memory = 8 * 1024**3  # 8GB
        mock_props.return_value = mock_device_props

        device_info = self.detector.detect_device()

        self.assertEqual(device_info.device_type, "cuda")
        self.assertEqual(device_info.device_name, "cuda")
        self.assertEqual(device_info.memory_gb, 8.0)
        self.assertTrue(device_info.supports_flash_attention)
        self.assertTrue(device_info.supports_quantization)

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_detect_device_mps(self, mock_mps_available, mock_cuda_available):
        """Test device detection when MPS is available."""
        device_info = self.detector.detect_device()

        self.assertEqual(device_info.device_type, "mps")
        self.assertEqual(device_info.device_name, "mps")
        self.assertEqual(device_info.memory_gb, 8.0)
        self.assertFalse(device_info.supports_flash_attention)
        self.assertFalse(device_info.supports_quantization)

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_detect_device_cpu(self, mock_mps_available, mock_cuda_available):
        """Test device detection when only CPU is available."""
        device_info = self.detector.detect_device()

        self.assertEqual(device_info.device_type, "cpu")
        self.assertEqual(device_info.device_name, "cpu")
        self.assertEqual(device_info.memory_gb, 4.0)
        self.assertFalse(device_info.supports_flash_attention)
        self.assertFalse(device_info.supports_quantization)

    def test_get_device_memory_cuda(self):
        """Test getting CUDA device memory."""
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_device_props = Mock()
            mock_device_props.total_memory = 16 * 1024**3  # 16GB
            mock_props.return_value = mock_device_props

            memory = self.detector.get_device_memory("cuda")
            self.assertEqual(memory, 16.0)

    def test_get_device_memory_mps(self):
        """Test getting MPS device memory."""
        memory = self.detector.get_device_memory("mps")
        self.assertEqual(memory, 8.0)

    def test_get_device_memory_cpu(self):
        """Test getting CPU device memory."""
        memory = self.detector.get_device_memory("cpu")
        self.assertEqual(memory, 4.0)


class TestDeviceService(UnitTestCase):
    """Test DeviceService implementation."""

    def setUp(self):
        super().setUp()
        self.service = DeviceService()

    @patch('torch.backends.mps.is_available', return_value=True)
    def test_get_optimal_device_mps(self, mock_mps_available):
        """Test getting optimal device when MPS is available."""
        device_info = self.service.get_optimal_device()

        self.assertEqual(device_info.device_type, "mps")
        self.assertEqual(device_info.device_name, "mps")

    def test_validate_model_compatibility_safe_model(self):
        """Test model compatibility validation with safe model."""
        device_info = DeviceInfo(
            device_type="mps",
            device_name="mps",
            torch_dtype="float16",
            memory_gb=8.0,
            supports_flash_attention=False,
            supports_quantization=False
        )

        # Safe model should not be changed
        result = self.service.validate_model_compatibility(device_info, "microsoft/Phi-3-mini-4k-instruct")
        self.assertEqual(result.device_type, "mps")

    def test_validate_model_compatibility_quantized_model(self):
        """Test model compatibility validation with quantized model."""
        device_info = DeviceInfo(
            device_type="mps",
            device_name="mps",
            torch_dtype="float16",
            memory_gb=8.0,
            supports_flash_attention=False,
            supports_quantization=False
        )

        # Quantized model should fallback to CPU
        result = self.service.validate_model_compatibility(device_info, "model-4bit")
        self.assertEqual(result.device_type, "cpu")


if __name__ == '__main__':
    unittest.main()
