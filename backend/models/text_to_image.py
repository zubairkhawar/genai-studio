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

            try:
                self.pipeline.enable_attention_slicing()
            except Exception:
                pass

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

        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(self.device).manual_seed(seed) if seed else None
            )

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