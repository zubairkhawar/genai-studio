import torch
import os
import asyncio
import pathlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from diffusers import DiffusionPipeline, StableVideoDiffusionPipeline
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import shutil
from .animatediff_generator import AnimateDiffGenerator
from .kandinsky_generator import KandinskyGenerator

class VideoGenerator:
    """Text-to-video generation using various models"""
    
    def __init__(self, gpu_info: Dict[str, Any]):
        self.gpu_info = gpu_info
        self.device = gpu_info["device"]
        self.models = {}
        self.available_models = {
            "animatediff": {
                "name": "AnimateDiff",
                "description": "Perfect for GIF generation - creates short, looping animations from text",
                "max_duration": 2,
                "resolution": "512x512",
                "type": "text2vid",
                "workflow": False,
                "features": [
                    "GIF generation",
                    "Looping animations",
                    "Memory efficient",
                    "Fast inference",
                    "M1 compatible"
                ]
            },
            "kandinsky": {
                "name": "Kandinsky 2.2",
                "description": "High-quality image generation with artistic style",
                "max_duration": 0,
                "resolution": "512x512",
                "type": "text2img",
                "workflow": False,
                "features": [
                    "Artistic style",
                    "High quality",
                    "Different from SD",
                    "Memory efficient"
                ]
            },
            "stable-diffusion": {
                "name": "Stable Diffusion",
                "description": "Text-to-image model used for video generation pipeline",
                "max_duration": 0,
                "resolution": "512x512",
                "type": "text2img",
                "workflow": False,
                "features": [
                    "Text-to-image generation",
                    "High quality images",
                    "Local-first inference"
                ]
            }
        }
    
    async def load_default_models(self):
        """Load default models - only called when explicitly requested"""
        try:
            # Check if models exist locally before attempting to load
            sd_path = pathlib.Path("../models/image/stable-diffusion")
            
            if sd_path.exists():
                print("Local Stable Diffusion model found...")
            else:
                print("No local models found. Use 'Download Models' to download them first.")
        except Exception as e:
            print(f"Warning: Could not load default video model: {e}")
    
    async def load_model(self, model_name: str) -> bool:
        """Load a specific model"""
        try:
            if model_name in self.models:
                return True
            
            print(f"Loading video model: {model_name}")
            
            if model_name == "animatediff":
                # Initialize AnimateDiff generator
                try:
                    print(f"Loading AnimateDiff generator...")
                    animatediff = AnimateDiffGenerator(
                        device=self.device,
                        dtype=torch.float16 if self.device != "cpu" else torch.float32
                    )
                    
                    success = await animatediff.load_model()
                    
                    if success:
                        print("✅ Successfully loaded AnimateDiff")
                        pipe = {"generator": True, "name": "animatediff", "generator_instance": animatediff}
                    else:
                        print("❌ Failed to load AnimateDiff, using placeholder")
                        pipe = {"placeholder": True, "name": "animatediff"}
                        
                except Exception as e:
                    print(f"Could not load AnimateDiff: {e}")
                    pipe = {"placeholder": True, "name": "animatediff"}
                    
            elif model_name == "kandinsky":
                # Initialize Kandinsky generator
                try:
                    print(f"Loading Kandinsky generator...")
                    kandinsky = KandinskyGenerator(
                        device=self.device,
                        dtype=torch.float16 if self.device != "cpu" else torch.float32
                    )
                    
                    success = await kandinsky.load_model()
                    
                    if success:
                        print("✅ Successfully loaded Kandinsky")
                        pipe = {"generator": True, "name": "kandinsky", "generator_instance": kandinsky}
                    else:
                        print("❌ Failed to load Kandinsky, using placeholder")
                        pipe = {"placeholder": True, "name": "kandinsky"}
                        
                except Exception as e:
                    print(f"Could not load Kandinsky: {e}")
                    pipe = {"placeholder": True, "name": "kandinsky"}
                    
            elif model_name == "stable-diffusion":
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
                    success = await sd_generator.load_model(sd_path)
                    
                    if success:
                        print("✅ Successfully loaded Stable Diffusion")
                        pipe = {"generator": True, "name": "stable-diffusion", "generator_instance": sd_generator}
                    else:
                        print("❌ Failed to load Stable Diffusion, using placeholder")
                        pipe = {"placeholder": True, "name": "stable-diffusion"}
                        
                except Exception as e:
                    print(f"Could not load Stable Diffusion: {e}")
                    pipe = {"placeholder": True, "name": "stable-diffusion"}
                    
                
            else:
                raise ValueError(f"Unknown model: {model_name}. Supported models: animatediff, kandinsky, stable-diffusion")
            
            self.models[model_name] = pipe
            print(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        available = []
        for model_id, info in self.available_models.items():
            # Check if model files exist on disk
            if model_id == "stable-diffusion":
                model_path = pathlib.Path(f"../models/image/{model_id}")
            elif model_id == "animatediff":
                # AnimateDiff has its own motion adapter files
                model_path = pathlib.Path(f"../models/video/animatediff")
            elif model_id == "kandinsky":
                # Kandinsky has its own directory
                model_path = pathlib.Path(f"../models/image/kandinsky")
            else:
                model_path = pathlib.Path(f"../models/video/{model_id}")
            
            # Always include the model, but check if it's downloaded
            size_gb = 0
            if model_path.exists():
                # Check for actual model weight files (not just config files)
                weight_files = list(model_path.rglob("*.safetensors")) + list(model_path.rglob("*.bin")) + list(model_path.rglob("*.pt")) + list(model_path.rglob("*.pth"))
                if len(weight_files) > 0:
                    # Calculate total model size
                    total_size = sum(f.stat().st_size for f in weight_files if f.is_file())
                    size_gb = total_size / (1024 * 1024 * 1024)
            
            available.append({
                "id": model_id,
                "name": info["name"],
                "description": info["description"],
                "max_duration": info["max_duration"],
                "resolution": info["resolution"],
                "size_gb": round(size_gb, 2) if size_gb > 0 else 0,
                "loaded": model_id in self.models
            })
        return available
    
    async def generate(self, prompt: str, model_name: str, duration: int = 5, 
                      output_format: str = "mp4", image_input: Optional[Union[str, Image.Image]] = None,
                      progress_callback: Optional[callable] = None) -> str:
        """Generate video from text prompt or image input"""
        try:
            # Ensure model is loaded
            if model_name not in self.models:
                await self.load_model(model_name)
            
            if model_name not in self.models:
                raise RuntimeError(f"Could not load model: {model_name}")
            
            # Generate video using the appropriate method
            if model_name == "animatediff":
                return await self._generate_with_animatediff(prompt, duration, output_format, progress_callback)
            elif model_name == "kandinsky":
                return await self._generate_with_kandinsky(prompt, duration, output_format, progress_callback)
            elif model_name == "stable-diffusion":
                return await self._generate_with_stable_diffusion(prompt, duration, output_format, progress_callback)
            elif model_name == "stable-video-diffusion":
                return await self._generate_with_pipeline(prompt, duration, output_format, image_input, progress_callback)
            else:
                raise ValueError(f"Unsupported model: {model_name}. Supported models: animatediff, kandinsky, stable-diffusion, stable-video-diffusion")
                
        except Exception as e:
            raise RuntimeError(f"Video generation failed: {e}")

    async def generate_image(self, prompt: str, model_name: str,
                             progress_callback: Optional[callable] = None) -> str:
        """Generate a single image using the specified model and save to outputs directory"""
        try:
            # Ensure model is loaded
            if model_name not in self.models:
                await self.load_model(model_name)

            if model_name not in self.models:
                raise RuntimeError(f"Could not load model: {model_name}")

            if model_name == "stable-diffusion":
                # Reuse Stable Diffusion image path
                if progress_callback:
                    progress_callback(10, "Preparing Stable Diffusion...")
                pipe = self.models["stable-diffusion"]
                if isinstance(pipe, dict) and pipe.get("generator"):
                    generator = pipe.get("generator_instance")
                    if progress_callback:
                        progress_callback(40, "Generating image...")
                    image = await generator.generate_image(
                        prompt=prompt,
                        width=512,
                        height=512,
                        num_inference_steps=20,
                        guidance_scale=7.5
                    )
                    if progress_callback:
                        progress_callback(80, "Saving image...")
                    output_dir = pathlib.Path("../outputs")
                    output_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"sd_image_{timestamp}.png"
                    output_path = output_dir / output_filename
                    image.save(output_path)
                    if progress_callback:
                        progress_callback(100, "Image generation complete!")
                    return str(output_path)
                # Placeholder if generator missing
                return await self._generate_placeholder(prompt, 1, "image", palette=(168, 85, 247))

            elif model_name == "kandinsky":
                # Use Kandinsky generator to create a single image and save PNG
                if progress_callback:
                    progress_callback(10, "Preparing Kandinsky...")
                pipe = self.models["kandinsky"]
                if isinstance(pipe, dict) and pipe.get("generator"):
                    generator = pipe.get("generator_instance")
                    if progress_callback:
                        progress_callback(40, "Generating image...")
                    image = await generator.generate_image(prompt=prompt)
                    if progress_callback:
                        progress_callback(80, "Saving image...")
                    output_dir = pathlib.Path("../outputs")
                    output_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"kandinsky_image_{timestamp}.png"
                    output_path = output_dir / output_filename
                    image.save(output_path)
                    if progress_callback:
                        progress_callback(100, "Image generation complete!")
                    return str(output_path)
                # Placeholder
                return await self._generate_placeholder(prompt, 1, "image", palette=(168, 85, 247))

            else:
                raise ValueError(f"Unsupported image model: {model_name}. Supported: stable-diffusion, kandinsky")

        except Exception as e:
            raise RuntimeError(f"Image generation failed: {e}")
    
    async def _generate_with_animatediff(self, prompt: str, duration: int, output_format: str, 
                                       progress_callback: Optional[callable] = None) -> str:
        """Generate video using AnimateDiff"""
        try:
            pipe = self.models["animatediff"]
            
            # Check if this is a placeholder model
            if isinstance(pipe, dict) and pipe.get("placeholder"):
                print("Using placeholder for AnimateDiff generation")
                return await self._generate_placeholder(prompt, duration, "gif", palette=(34, 197, 94))
            
            # Check if this is a generator model
            if isinstance(pipe, dict) and pipe.get("generator"):
                generator = pipe.get("generator_instance")
                if generator:
                    print("Using AnimateDiff for generation")
                    
                    if progress_callback:
                        progress_callback(25, "Generating GIF with AnimateDiff...")
                    
                    output_path = await generator.generate_video(
                        prompt=prompt,
                        num_frames=min(duration * 8, 16),  # 8 fps, max 16 frames
                        progress_callback=progress_callback
                    )
                    
                    return output_path
                else:
                    print("AnimateDiff generator instance not available, using placeholder")
                    return await self._generate_placeholder(prompt, duration, "gif", palette=(34, 197, 94))
            
            # Fallback to placeholder
            print("Using placeholder for AnimateDiff generation")
            return await self._generate_placeholder(prompt, duration, "gif", palette=(34, 197, 94))
            
        except Exception as e:
            raise RuntimeError(f"AnimateDiff generation failed: {e}")
    
    async def _generate_with_kandinsky(self, prompt: str, duration: int, output_format: str, 
                                     progress_callback: Optional[callable] = None) -> str:
        """Generate image using Kandinsky (for image generation)"""
        try:
            pipe = self.models["kandinsky"]
            
            # Check if this is a placeholder model
            if isinstance(pipe, dict) and pipe.get("placeholder"):
                print("Using placeholder for Kandinsky generation")
                return await self._generate_placeholder(prompt, duration, output_format, palette=(168, 85, 247))
            
            # Check if this is a generator model
            if isinstance(pipe, dict) and pipe.get("generator"):
                generator = pipe.get("generator_instance")
                if generator:
                    print("Using Kandinsky for image generation")
                    
                    if progress_callback:
                        progress_callback(25, "Generating image with Kandinsky...")
                    
                    # Generate single image
                    image = await generator.generate_image(prompt=prompt)
                    
                    # Convert to video by duplicating the frame
                    frames = [image] * max(6, duration * 6)  # 6 fps
                    
                    if progress_callback:
                        progress_callback(80, "Converting image to video...")
                    
                    output_path = await self._save_video_frames(frames, output_format)
                    
                    return output_path
                else:
                    print("Kandinsky generator instance not available, using placeholder")
                    return await self._generate_placeholder(prompt, duration, output_format, palette=(168, 85, 247))
            
            # Fallback to placeholder
            print("Using placeholder for Kandinsky generation")
            return await self._generate_placeholder(prompt, duration, output_format, palette=(168, 85, 247))
            
        except Exception as e:
            raise RuntimeError(f"Kandinsky generation failed: {e}")
    
    async def _generate_with_stable_diffusion(self, prompt: str, duration: int, output_format: str, 
                                            progress_callback: Optional[callable] = None) -> str:
        """Generate image using Stable Diffusion"""
        try:
            pipe = self.models["stable-diffusion"]
            
            # Check if this is a placeholder model
            if isinstance(pipe, dict) and pipe.get("placeholder"):
                print("Using placeholder for Stable Diffusion generation")
                return await self._generate_placeholder(prompt, duration, "image", palette=(168, 85, 247))
            
            # Check if this is a generator model
            if isinstance(pipe, dict) and pipe.get("generator"):
                generator = pipe.get("generator_instance")
                if generator:
                    print("Using Stable Diffusion for image generation")
                    
                    if progress_callback:
                        progress_callback(25, "Generating image with Stable Diffusion...")
                    
                    # Generate image
                    image = await generator.generate_image(
                        prompt=prompt,
                        width=512,
                        height=512,
                        num_inference_steps=20,
                        guidance_scale=7.5
                    )
                    
                    if progress_callback:
                        progress_callback(75, "Saving generated image...")
                    
                    # Save the image
                    output_dir = pathlib.Path("../outputs")
                    output_dir.mkdir(exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"sd_image_{timestamp}.png"
                    output_path = output_dir / output_filename
                    
                    image.save(output_path)
                    
                    if progress_callback:
                        progress_callback(100, "Image generation complete!")
                    
                    print(f"✅ Generated image saved to: {output_path}")
                    return str(output_path)
                else:
                    print("Stable Diffusion generator instance not available, using placeholder")
                    return await self._generate_placeholder(prompt, duration, "image", palette=(168, 85, 247))
            
            # Fallback to placeholder
            print("Using placeholder for Stable Diffusion generation")
            return await self._generate_placeholder(prompt, duration, "image", palette=(168, 85, 247))
            
        except Exception as e:
            raise RuntimeError(f"Stable Diffusion generation failed: {e}")
    
    async def _generate_with_pipeline(self, prompt: str, duration: int, output_format: str, 
                                    image_input: Optional[Union[str, Image.Image]] = None,
                                    progress_callback: Optional[callable] = None) -> str:
        """Generate video using the text-to-video pipeline"""
        try:
            pipe = self.models["stable-video-diffusion"]
            
            # Check if this is a placeholder model
            if isinstance(pipe, dict) and pipe.get("placeholder"):
                print("Using placeholder for video generation")
                return await self._generate_placeholder(prompt, duration, output_format, palette=(56, 189, 248))
            
            # Check if this is a pipeline model
            if isinstance(pipe, dict) and pipe.get("pipeline"):
                pipeline = pipe.get("pipeline_instance")
                if pipeline:
                    print("Using Text-to-Video pipeline for generation")
                    
                    if image_input is not None:
                        # Image-to-video generation
                        print("Generating video from provided image...")
                        if progress_callback:
                            progress_callback(25, "Generating video from image...")
                        output_path = await pipeline.generate_video_from_image(
                            image_input=image_input,
                            prompt=prompt,
                            video_duration=duration,
                            video_seed=42
                        )
                    else:
                        # Text-to-video generation
                        print("Generating video from text prompt...")
                        if progress_callback:
                            progress_callback(25, "Generating video from text...")
                        output_path = await pipeline.generate_video_from_text(
                            prompt=prompt,
                            video_duration=duration,
                            image_seed=42,
                            video_seed=42
                        )
                    
                    return output_path
                else:
                    print("Pipeline instance not available, using placeholder")
                    return await self._generate_placeholder(prompt, duration, output_format, palette=(56, 189, 248))
            
            # Fallback to placeholder
            print("Using placeholder for video generation")
            return await self._generate_placeholder(prompt, duration, output_format, palette=(56, 189, 248))
            
        except Exception as e:
            raise RuntimeError(f"Video generation failed: {e}")
    
    
    async def _generate_placeholder(self, prompt: str, duration: int, output_format: str, palette=(56,189,248)) -> str:
        """Generate a lightweight placeholder video with animated gradient and prompt overlay.
        This enables end-to-end flow without heavy model weights.
        """
        try:
            width, height = 192, 192
            fps = 6
            num_frames = max(6, min(duration * fps, 60))
            frames: List[Image.Image] = []
            r, g, b = palette
            for i in range(num_frames):
                # Animated background color shift
                factor = (i / num_frames)
                bg = (int(r*(0.6+0.4*factor)) % 255, int(g*(0.6+0.4*(1-factor)) % 255), int(b*(0.6+0.4*abs(0.5-factor)) % 255))
                img = Image.new('RGB', (width, height), color=bg)
                draw = ImageDraw.Draw(img)
                # Simple progress bar
                bar_w = int((i+1)/num_frames * (width-40))
                draw.rectangle([(20, height-30), (20+bar_w, height-20)], fill=(255,255,255))
                # Prompt overlay (truncated)
                text = (prompt or "Placeholder video")[0:60]
                draw.text((20, 20), text, fill=(255,255,255))
                frames.append(img)
            return await self._save_video_frames(frames, output_format)
        except Exception as e:
            raise RuntimeError(f"Placeholder generation failed: {e}")
    
    async def _save_video_frames(self, frames: List[Image.Image], output_format: str) -> str:
        """Save video frames to file"""
        try:
            # Create output directory
            os.makedirs("../outputs/videos", exist_ok=True)
            
            # Generate output filename
            import uuid
            filename = f"video_{uuid.uuid4().hex[:8]}.{output_format}"
            output_path = f"../outputs/videos/{filename}"
            
            # Convert PIL images to numpy arrays
            frame_arrays = []
            for frame in frames:
                frame_array = np.array(frame)
                frame_arrays.append(frame_array)
            
            # Get video dimensions
            height, width, channels = frame_arrays[0].shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 6.0, (width, height))
            
            # Write frames
            for frame_array in frame_arrays:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to save video: {e}")
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory"""
        try:
            if model_name in self.models:
                
                del self.models[model_name]
                torch.cuda.empty_cache() if self.device != "cpu" else None
                return True
            return False
        except Exception as e:
            print(f"Error unloading model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name in self.available_models:
            info = self.available_models[model_name].copy()
            info["loaded"] = model_name in self.models
            return info
        return None
