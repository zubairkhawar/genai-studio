"""
Stable Video Diffusion Workflow Implementation

This module implements the SVD workflow based on the ComfyUI workflow provided.
It includes proper image-to-video generation with frame interpolation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import tempfile
from typing import List, Optional, Dict, Any, Tuple
from diffusers import StableVideoDiffusionPipeline, AutoencoderKL
from diffusers.utils import load_image, export_to_video
import logging

logger = logging.getLogger(__name__)

class SVDWorkflow:
    """Stable Video Diffusion workflow implementation"""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.pipeline = None
        self.vae = None
        self.is_loaded = False
        
        # Workflow parameters (from the ComfyUI workflow)
        self.config = {
            "width": 1024,
            "height": 576,
            "num_frames": 25,
            "num_inference_steps": 12,
            "guidance_scale": 2.0,
            "min_guidance_scale": 0.02,
            "motion_bucket_id": 100,
            "noise_aug_strength": 0.02,
            "decode_chunk_size": 8,
            "frame_rate": 20,
            "interpolation_factor": 2,  # RIFE interpolation
            "freeu_enabled": True,
            "freeu_params": {
                "b1": 1.3,
                "b2": 1.4,
                "s1": 0.9,
                "s2": 0.2
            }
        }
    
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the SVD model"""
        try:
            if self.is_loaded:
                return True
            
            logger.info("Loading Stable Video Diffusion model...")
            
            # Try local path first
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading SVD from local path: {model_path}")
                self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None,
                    local_files_only=True
                )
            else:
                # Fallback to Hugging Face
                logger.info("Loading SVD from Hugging Face...")
                self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion",
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None
                )
            
            # Move to device and enable optimizations
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_model_cpu_offload()
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            # Enable VAE slicing for memory efficiency
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
            
            self.is_loaded = True
            logger.info("✅ SVD model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SVD model: {e}")
            return False
    
    def _prepare_image(self, image_input: Any) -> Image.Image:
        """Prepare input image for SVD"""
        if isinstance(image_input, str):
            # Load from file path
            if os.path.exists(image_input):
                image = Image.open(image_input).convert("RGB")
            else:
                # Create a placeholder image
                image = self._create_placeholder_image()
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            # Create a placeholder image
            image = self._create_placeholder_image()
        
        # Resize to SVD requirements (1024x576)
        image = image.resize((self.config["width"], self.config["height"]), Image.Resampling.LANCZOS)
        return image
    
    def _create_placeholder_image(self) -> Image.Image:
        """Create a placeholder image for testing"""
        image = Image.new('RGB', (self.config["width"], self.config["height"]), color='white')
        draw = ImageDraw.Draw(image)
        
        # Add some visual elements
        draw.rectangle([50, 50, self.config["width"]-50, self.config["height"]-50], 
                      outline='blue', width=5)
        draw.text((100, 100), "SVD Placeholder", fill='blue')
        
        return image
    
    def _apply_freeu(self, model, enabled: bool = True):
        """Apply FreeU enhancement to the model"""
        if not enabled or not hasattr(model, 'enable_freeu'):
            return model
        
        try:
            params = self.config["freeu_params"]
            model.enable_freeu(
                s1=params["s1"],
                s2=params["s2"],
                b1=params["b1"],
                b2=params["b2"]
            )
            logger.info("FreeU enhancement applied")
        except Exception as e:
            logger.warning(f"Could not apply FreeU: {e}")
        
        return model
    
    def _interpolate_frames(self, frames: List[Image.Image], factor: int = 2) -> List[Image.Image]:
        """Apply frame interpolation using simple linear interpolation"""
        if factor <= 1 or len(frames) < 2:
            return frames
        
        interpolated_frames = []
        
        for i in range(len(frames) - 1):
            # Add original frame
            interpolated_frames.append(frames[i])
            
            # Add interpolated frames
            for j in range(1, factor):
                alpha = j / factor
                
                # Simple linear interpolation between frames
                frame1 = np.array(frames[i])
                frame2 = np.array(frames[i + 1])
                
                # Blend frames
                blended = (frame1 * (1 - alpha) + frame2 * alpha).astype(np.uint8)
                interpolated_frames.append(Image.fromarray(blended))
        
        # Add last frame
        interpolated_frames.append(frames[-1])
        
        return interpolated_frames
    
    async def generate_video(
        self, 
        image_input: Any,
        prompt: str = "",
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        motion_bucket_id: Optional[int] = None,
        noise_aug_strength: Optional[float] = None,
        seed: Optional[int] = None
    ) -> str:
        """Generate video from image using SVD workflow"""
        
        if not self.is_loaded:
            raise RuntimeError("SVD model not loaded. Call load_model() first.")
        
        try:
            # Prepare parameters
            num_frames = num_frames or self.config["num_frames"]
            num_inference_steps = num_inference_steps or self.config["num_inference_steps"]
            guidance_scale = guidance_scale or self.config["guidance_scale"]
            motion_bucket_id = motion_bucket_id or self.config["motion_bucket_id"]
            noise_aug_strength = noise_aug_strength or self.config["noise_aug_strength"]
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Prepare input image
            image = self._prepare_image(image_input)
            logger.info(f"Prepared input image: {image.size}")
            
            # Apply FreeU enhancement
            self._apply_freeu(self.pipeline.unet, self.config["freeu_enabled"])
            
            # Generate video frames
            logger.info("Generating video frames...")
            video_frames = self.pipeline(
                image,
                decode_chunk_size=self.config["decode_chunk_size"],
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                min_guidance_scale=guidance_scale,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                generator=torch.Generator(device=self.device).manual_seed(seed or 42)
            ).frames[0]
            
            logger.info(f"Generated {len(video_frames)} frames")
            
            # Apply frame interpolation if enabled
            if self.config["interpolation_factor"] > 1:
                logger.info("Applying frame interpolation...")
                video_frames = self._interpolate_frames(
                    video_frames, 
                    self.config["interpolation_factor"]
                )
                logger.info(f"Interpolated to {len(video_frames)} frames")
            
            # Save video
            output_path = await self._save_video_frames(video_frames)
            logger.info(f"Video saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise RuntimeError(f"SVD video generation failed: {e}")
    
    async def _save_video_frames(self, frames: List[Image.Image]) -> str:
        """Save video frames to MP4 file"""
        try:
            # Create output directory
            os.makedirs("../outputs/videos", exist_ok=True)
            
            # Generate output filename
            import uuid
            filename = f"svd_video_{uuid.uuid4().hex[:8]}.mp4"
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
            out = cv2.VideoWriter(
                output_path, 
                fourcc, 
                self.config["frame_rate"], 
                (width, height)
            )
            
            # Write frames
            for frame_array in frame_arrays:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise RuntimeError(f"Failed to save video: {e}")
    
    def unload_model(self) -> bool:
        """Unload the model to free memory"""
        try:
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            logger.info("SVD model unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading SVD model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "Stable Video Diffusion",
            "description": "High-quality image-to-video generation",
            "loaded": self.is_loaded,
            "device": self.device,
            "dtype": str(self.dtype),
            "config": self.config,
            "capabilities": [
                "image-to-video",
                "frame interpolation",
                "customizable parameters",
                "memory efficient"
            ]
        }
    
    def update_config(self, **kwargs):
        """Update workflow configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            elif key == "freeu_params" and isinstance(value, dict):
                self.config["freeu_params"].update(value)
        
        logger.info(f"Updated config: {kwargs}")
