"""
Image Generator for text-to-image generation
Handles Stable Diffusion and other image generation models
"""

import os
import pathlib
import torch
import asyncio
from typing import Dict, List, Optional, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime

class ImageGenerator:
    """Text-to-image generation using various models"""
    
    def __init__(self, gpu_info: Dict[str, Any]):
        self.gpu_info = gpu_info
        self.device = gpu_info["device"]
        self.models = {}
        self.available_models = {
            "stable-diffusion": {
                "id": "stable-diffusion",
                "name": "Stable Diffusion v1.5",
                "description": "High-quality text-to-image generation with customizable settings",
                "max_resolution": "2048x2048",
                "type": "diffusion",
                "features": [
                    "Photorealistic images",
                    "Artistic styles",
                    "Customizable resolution",
                    "Advanced quality controls",
                    "Manual settings support"
                ]
            }
        }
        
        # Initialize default model
        try:
            # Check if models exist locally before attempting to load
            sd_path = pathlib.Path("../models/image/stable-diffusion")
            
            if sd_path.exists():
                print("Local Stable Diffusion model found...")
            else:
                print("No local models found. Use 'Download Models' to download them first.")
        except Exception as e:
            print(f"Warning: Could not load default image model: {e}")
    
    async def load_model(self, model_name: str) -> bool:
        """Load a specific image model"""
        try:
            if model_name in self.models:
                return True
            
            print(f"Loading image model: {model_name}")
            
            if model_name == "stable-diffusion":
                # Initialize Stable Diffusion text-to-image generator
                try:
                    print(f"Loading Stable Diffusion generator...")
                    from .text_to_image import TextToImageGenerator
                    
                    sd_generator = TextToImageGenerator(
                        device=self.device,
                        dtype=torch.float32 if self.device == "mps" else (torch.float16 if self.device != "cpu" else torch.float32)
                    )
                    
                    # Try to load from local directory first
                    sd_path = "../models/image/stable-diffusion"
                    success = sd_generator.load_model(sd_path)
                    
                    if success:
                        print("✅ Successfully loaded Stable Diffusion")
                        pipe = {"generator": True, "name": "stable-diffusion", "generator_instance": sd_generator}
                    else:
                        print("❌ Failed to load Stable Diffusion")
                        raise RuntimeError("Failed to load Stable Diffusion model")
                        
                except Exception as e:
                    print(f"Could not load Stable Diffusion: {e}")
                    raise RuntimeError(f"Failed to load Stable Diffusion: {e}")
                    
            else:
                raise ValueError(f"Unsupported image model: {model_name}")
            
            self.models[model_name] = pipe
            return True
            
        except Exception as e:
            print(f"Error loading image model {model_name}: {e}")
            return False
    
    async def generate_image(self, prompt: str, model_name: str,
                           width: int = 1024, height: int = 1024,
                           num_inference_steps: int = 50,
                           guidance_scale: float = 9.0,
                           progress_callback: Optional[callable] = None) -> str:
        """Generate a single image with Stable Diffusion"""
        try:
            # Ensure model is loaded
            if model_name not in self.models:
                await self.load_model(model_name)

            if model_name not in self.models:
                raise RuntimeError(f"Could not load model: {model_name}")

            if model_name == "stable-diffusion":
                if progress_callback:
                    progress_callback(10, "Preparing Stable Diffusion...")
                
                pipe = self.models.get("stable-diffusion")
                if not pipe or not pipe.get("generator"):
                    raise RuntimeError("Stable Diffusion not loaded correctly")

                generator = pipe["generator_instance"]
                
                if progress_callback:
                    progress_callback(40, "Generating image...")

                # Generate image using asyncio.to_thread for the synchronous method
                print(f"DEBUG: About to call generator.generate_image")
                print(f"DEBUG: Generator type: {type(generator)}")
                print(f"DEBUG: Generator has generate_image: {hasattr(generator, 'generate_image')}")
                
                image = await asyncio.to_thread(
                    generator.generate_image,
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                
                print(f"DEBUG: Image type after asyncio.to_thread: {type(image)}")
                print(f"DEBUG: Image has save method: {hasattr(image, 'save')}")
                
                if progress_callback:
                    progress_callback(80, "Saving image...")
                
                output_dir = pathlib.Path("../outputs")
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"sd_image_{timestamp}.png"
                output_path = output_dir / output_filename
                
                # Save the image
                image.save(output_path)
                
                if progress_callback:
                    progress_callback(100, "Image generation complete!")
                
                return str(output_path)
            else:
                raise ValueError(f"Unsupported image model: {model_name}. Supported: stable-diffusion")

        except Exception as e:
            raise RuntimeError(f"Image generation failed: {e}")
    
    async def _download_model_fallback(self):
        """Download Stable Diffusion model using diffusers if not available locally"""
        try:
            print("Downloading Stable Diffusion model from Hugging Face...")
            from diffusers import StableDiffusionPipeline
            
            # Download to local directory
            local_path = "../models/image/stable-diffusion"
            os.makedirs(local_path, exist_ok=True)
            
            # Download model in background thread
            pipeline = await asyncio.to_thread(
                StableDiffusionPipeline.from_pretrained,
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                cache_dir=local_path
            )
            
            # Save to local directory
            await asyncio.to_thread(pipeline.save_pretrained, local_path)
            
            # Now try to load it properly
            await self.load_model("stable-diffusion")
            
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise RuntimeError(f"Could not download or load Stable Diffusion model: {e}")
    
    async def _generate_placeholder(self, prompt: str) -> str:
        """Generate a placeholder image"""
        try:
            # Create output directory
            os.makedirs("../outputs", exist_ok=True)
            
            # Generate placeholder image
            width, height = 512, 512
            img = Image.new('RGB', (width, height), color=(168, 85, 247))  # Purple background
            draw = ImageDraw.Draw(img)
            
            # Add prompt text
            text = (prompt or "Placeholder image")[:60]
            try:
                # Try to use a default font
                font = ImageFont.load_default()
            except:
                font = None
            
            # Draw text in the center
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            # Save the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"placeholder_image_{timestamp}.png"
            output_path = f"../outputs/{filename}"
            img.save(output_path)
            
            print(f"✅ Generated placeholder image saved to: {output_path}")
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate placeholder image: {e}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available image models with download status"""
        models = []
        for model_id, model_info in self.available_models.items():
            model_status = model_info.copy()
            
            # Check if model is actually downloaded
            if model_id == "stable-diffusion":
                sd_path = pathlib.Path("../models/image/stable-diffusion")
                if sd_path.exists():
                    # Check for essential files
                    required_files = ["config.json", "model_index.json", "v1-inference.yaml"]
                    required_dirs = ["text_encoder", "unet", "vae", "scheduler", "tokenizer"]
                    
                    has_files = all((sd_path / f).exists() for f in required_files)
                    has_dirs = all((sd_path / d).exists() for d in required_dirs)
                    
                    if has_files and has_dirs:
                        model_status["status"] = "available"
                        model_status["size_gb"] = self._get_model_size(sd_path)
                    else:
                        model_status["status"] = "download"
                        model_status["size_gb"] = 0
                else:
                    model_status["status"] = "download"
                    model_status["size_gb"] = 0
            else:
                model_status["status"] = "download"
                model_status["size_gb"] = 0
            
            models.append(model_status)
        
        return models
    
    def _get_model_size(self, model_path: pathlib.Path) -> float:
        """Calculate model size in GB"""
        try:
            total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
            return round(total_size / (1024**3), 2)
        except Exception:
            return 0.0
