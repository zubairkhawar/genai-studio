import torch
import platform
import subprocess
import re
from typing import Dict, Any

class GPUDetector:
    """Detect and configure GPU support for both NVIDIA and AMD"""
    
    def __init__(self):
        self.gpu_info = self.detect_gpu()
    
    def detect_gpu(self) -> Dict[str, Any]:
        """Detect available GPU and return configuration"""
        gpu_info = {
            "type": "cpu",
            "device": "cpu",
            "cuda_available": False,
            "rocm_available": False,
            "memory_gb": 0,
            "name": "Unknown"
        }
        
        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            gpu_info.update({
                "type": "cuda",
                "device": "cuda",
                "cuda_available": True,
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "name": torch.cuda.get_device_name(0)
            })
            return gpu_info
        
        # Check for ROCm (AMD)
        try:
            import torch_rocm
            if torch_rocm.is_available():
                gpu_info.update({
                    "type": "rocm",
                    "device": "cuda",  # ROCm uses CUDA API
                    "rocm_available": True,
                    "memory_gb": self._get_rocm_memory(),
                    "name": self._get_rocm_gpu_name()
                })
                return gpu_info
        except ImportError:
            pass
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info.update({
                "type": "mps",
                "device": "mps",
                "memory_gb": self._get_mps_memory(),
                "name": "Apple Silicon GPU"
            })
            return gpu_info
        
        # Fallback to CPU
        gpu_info["memory_gb"] = self._get_cpu_memory()
        return gpu_info
    
    def _get_rocm_memory(self) -> float:
        """Get ROCm GPU memory in GB"""
        try:
            result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # Parse memory info from rocm-smi output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'GPU[0]' in line and 'vram' in line:
                        # Extract memory value
                        match = re.search(r'(\d+\.?\d*)\s*MB', line)
                        if match:
                            return float(match.group(1)) / 1024
        except Exception:
            pass
        return 8.0  # Default fallback
    
    def _get_rocm_gpu_name(self) -> str:
        """Get ROCm GPU name"""
        try:
            result = subprocess.run(['rocm-smi', '--showproductname'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Card series' in line:
                        return line.split(':')[1].strip()
            
            # Try alternative method for newer ROCm versions
            result = subprocess.run(['rocm-smi', '--showid'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'GPU[0]' in line:
                        # Extract GPU name from the line
                        parts = line.split()
                        for part in parts:
                            if '7900' in part or 'XTX' in part or 'RX' in part:
                                return f"AMD {part}"
        except Exception:
            pass
        return "AMD GPU"
    
    def _get_mps_memory(self) -> float:
        """Get MPS memory in GB (approximate)"""
        try:
            import psutil
            # MPS shares system memory, so we estimate based on available RAM
            total_memory = psutil.virtual_memory().total / 1024**3
            return min(total_memory * 0.5, 16.0)  # Use up to 50% of RAM, max 16GB
        except Exception:
            return 8.0
    
    def _get_cpu_memory(self) -> float:
        """Get CPU memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / 1024**3
        except Exception:
            return 8.0
    
    def get_device(self) -> str:
        """Get the appropriate device string for PyTorch"""
        return self.gpu_info["device"]
    
    def is_gpu_available(self) -> bool:
        """Check if any GPU is available"""
        return self.gpu_info["type"] != "cpu"
    
    def get_memory_gb(self) -> float:
        """Get available memory in GB"""
        return self.gpu_info["memory_gb"]
