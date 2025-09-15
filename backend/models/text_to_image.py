"""
Text-to-Image Generation Module

This module provides text-to-image generation capabilities using Stable Diffusion,
which can then be used as input for the SVD image-to-video pipeline.
"""

import torch
import os
from PIL import Image
from typing import Optional, Dict, Any
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import logging

logger = logging.getLogger(__name__)

class TextToImageGenerator:
    """Text-to-image generation using Stable Diffusion"""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.pipeline = None
        self.is_loaded = False
        
        # Configuration for SVD compatibility
        self.config = {
            "width": 1024,
            "height": 576,  # SVD expects 576x1024
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "scheduler": "DPMSolverMultistepScheduler"
        }
    
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the Stable Diffusion model"""
        try:
            if self.is_loaded:
                return True
            
            logger.info("Loading Stable Diffusion model for text-to-image...")
            
            # Try local path first
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading Stable Diffusion from local path: {model_path}")
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                    local_files_only=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                # Use a lightweight, fast model for text-to-image
                logger.info("Loading Stable Diffusion from Hugging Face...")
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",  # Fast and reliable
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            # Move to device and optimize
            self.pipeline = self.pipeline.to(self.device)
            
            # Use a faster scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Enable memory optimizations
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
            
            self.is_loaded = True
            logger.info("✅ Stable Diffusion model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion model: {e}")
            return False
    
    async def generate_image(
        self, 
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate image from text prompt"""
        
        if not self.is_loaded:
            raise RuntimeError("Stable Diffusion model not loaded. Call load_model() first.")
        
        try:
            # Set parameters
            width = width or self.config["width"]
            height = height or self.config["height"]
            num_inference_steps = num_inference_steps or self.config["num_inference_steps"]
            guidance_scale = guidance_scale or self.config["guidance_scale"]
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            logger.info(f"Generating image from prompt: '{prompt[:50]}...'")
            
            # Generate image
            with torch.autocast(self.device):
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(seed or 42)
                )
            
            image = result.images[0]
            
            # Ensure the image is in the correct format for SVD
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            logger.info(f"✅ Generated image: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise RuntimeError(f"Text-to-image generation failed: {e}")
    
    def unload_model(self) -> bool:
        """Unload the model to free memory"""
        try:
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            logger.info("Stable Diffusion model unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading Stable Diffusion model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "Stable Diffusion v1.5",
            "description": "Text-to-image generation for SVD input",
            "loaded": self.is_loaded,
            "device": self.device,
            "dtype": str(self.dtype),
            "config": self.config,
            "capabilities": [
                "text-to-image",
                "SVD-compatible output",
                "fast inference",
                "memory efficient"
            ]
        }
    
    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        
        logger.info(f"Updated config: {kwargs}")
