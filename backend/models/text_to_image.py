import os
import pathlib
import torch
from typing import Optional
from PIL import Image

class TextToImageGenerator:
    """Stable Diffusion text-to-image generator"""

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self.pipeline = None
        self.model_path = None

    def load_model(self, model_path: str) -> bool:
        """Load Stable Diffusion model from local path (sync)"""
        try:
            print(f"Loading Stable Diffusion from: {model_path}")
            if not pathlib.Path(model_path).exists():
                print(f"❌ Model path does not exist: {model_path}")
                return False

            from diffusers import StableDiffusionPipeline

            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)

            # Apply cross-platform optimizations
            try:
                self.pipeline.enable_attention_slicing()
                print("✅ Enabled attention slicing for SD")
            except Exception:
                pass
            
            # CUDA-specific optimizations
            if self.device == "cuda":
                import platform
                os_name = platform.system().lower()
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                print(f"Applying SD optimizations for {memory_gb:.1f}GB GPU on {os_name}")
                
                # Memory optimizations for different GPU sizes
                if memory_gb < 12:  # 8GB and lower GPUs
                    try:
                        self.pipeline.enable_cpu_offload()
                        print("✅ Enabled CPU offload for SD on low-VRAM GPU")
                    except Exception:
                        pass
                
                # Platform-specific optimizations
                if os_name == "windows":
                    torch.cuda.set_per_process_memory_fraction(0.9)
                    print("✅ Set Windows CUDA memory fraction to 90% for SD")
                elif os_name == "linux":
                    torch.cuda.empty_cache()
                    print("✅ Applied Linux CUDA optimizations for SD")

            self.model_path = model_path
            print("✅ Stable Diffusion loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading Stable Diffusion model: {e}")
            return False

    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024,
                       num_inference_steps: int = 50, guidance_scale: float = 9.0,
                       seed: Optional[int] = None) -> Image.Image:
        """Generate an image from text prompt (sync)"""
        if self.pipeline is None:
            raise RuntimeError("Stable Diffusion pipeline not loaded")

        print(f"🎨 Generating image: '{prompt[:50]}...' ({width}x{height})")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Clear memory before generation
        if self.device == "cuda":
            torch.cuda.empty_cache()
            import platform
            os_name = platform.system().lower()
            if os_name == "windows":
                torch.cuda.synchronize()
                print("✅ Synchronized CUDA on Windows before SD generation")
            elif os_name == "linux":
                torch.cuda.empty_cache()
                print("✅ Applied Linux CUDA memory management before SD generation")

        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(self.device).manual_seed(seed) if seed else None
            )

        # Clear memory after generation
        if self.device == "cuda":
            torch.cuda.empty_cache()
            import platform
            os_name = platform.system().lower()
            if os_name == "windows":
                torch.cuda.synchronize()
                print("✅ Synchronized CUDA on Windows after SD generation")
            elif os_name == "linux":
                torch.cuda.empty_cache()
                print("✅ Applied Linux CUDA cleanup after SD generation")

        image = result.images[0]
        if not isinstance(image, Image.Image):
            raise RuntimeError(f"Expected PIL.Image, got {type(image)}")

        print("✅ Image generated successfully")
        return image

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.pipeline is not None

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": str(self.dtype),
            "is_loaded": self.is_loaded()
        }