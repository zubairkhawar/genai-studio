"""
Text-to-Video Pipeline

This module combines text-to-image generation with SVD image-to-video generation
to create a complete text-to-video pipeline.
"""

import torch
import os
import asyncio
from PIL import Image
from typing import Optional, Dict, Any, Union
import logging
from .text_to_image import TextToImageGenerator
from .svd_workflow import SVDWorkflow

logger = logging.getLogger(__name__)

class TextToVideoPipeline:
    """Complete text-to-video pipeline: text → image → video"""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.text_to_image = None
        self.svd_workflow = None
        self.is_loaded = False
        
        # Pipeline configuration - Optimized for SVD
        self.config = {
            "image_generation": {
                "width": 192,  # Ultra-aggressive reduction for 8GB limit
                "height": 192,  # Ultra-aggressive reduction for 8GB limit
                "num_inference_steps": 10,  # Ultra-minimal for 8GB limit
                "guidance_scale": 7.5
            },
            "video_generation": {
                "num_frames": 2,  # Ultra-minimal frames for 8GB limit
                "num_inference_steps": 4,  # Ultra-minimal for 8GB limit
                "guidance_scale": 2.0,
                "motion_bucket_id": 100,
                "noise_aug_strength": 0.02
            }
        }
    
    async def load_models(self, 
                         text_to_image_path: Optional[str] = None,
                         svd_path: Optional[str] = None) -> bool:
        """Load both text-to-image and SVD models"""
        try:
            logger.info("Loading text-to-video pipeline models...")
            
            # Initialize text-to-image generator
            self.text_to_image = TextToImageGenerator(self.device, self.dtype)
            image_success = await self.text_to_image.load_model(text_to_image_path)
            
            # Initialize SVD workflow
            self.svd_workflow = SVDWorkflow(self.device, self.dtype)
            svd_success = await self.svd_workflow.load_model(svd_path)
            
            if image_success and svd_success:
                self.is_loaded = True
                logger.info("✅ Text-to-video pipeline loaded successfully")
                return True
            else:
                logger.error("Failed to load one or more models")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load text-to-video pipeline: {e}")
            return False
    
    async def generate_video_from_text(
        self,
        prompt: str,
        image_prompt: Optional[str] = None,
        video_duration: int = 4,
        image_seed: Optional[int] = None,
        video_seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate video from text prompt using the complete pipeline"""
        
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_models() first.")
        
        try:
            logger.info(f"Starting text-to-video generation: '{prompt[:50]}...'")
            
            # Step 1: Generate image from text
            image_prompt = image_prompt or prompt
            logger.info("Step 1: Generating image from text...")
            
            image = await self.text_to_image.generate_image(
                prompt=image_prompt,
                width=self.config["image_generation"]["width"],
                height=self.config["image_generation"]["height"],
                num_inference_steps=self.config["image_generation"]["num_inference_steps"],
                guidance_scale=self.config["image_generation"]["guidance_scale"],
                seed=image_seed
            )
            
            logger.info("✅ Image generated successfully")
            
            # Step 2: Generate video from image
            logger.info("Step 2: Generating video from image...")
            
            video_path = await self.svd_workflow.generate_video(
                image_input=image,
                prompt=prompt,
                num_frames=min(video_duration * 1, 2),  # 1 fps, max 2 frames
                num_inference_steps=self.config["video_generation"]["num_inference_steps"],
                guidance_scale=self.config["video_generation"]["guidance_scale"],
                motion_bucket_id=self.config["video_generation"]["motion_bucket_id"],
                noise_aug_strength=self.config["video_generation"]["noise_aug_strength"],
                seed=video_seed
            )
            
            logger.info("✅ Video generated successfully")
            return video_path
            
        except Exception as e:
            logger.error(f"Text-to-video generation failed: {e}")
            raise RuntimeError(f"Text-to-video pipeline failed: {e}")
    
    async def generate_video_from_image(
        self,
        image_input: Union[str, Image.Image],
        prompt: str = "",
        video_duration: int = 4,
        video_seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate video from existing image (direct SVD)"""
        
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_models() first.")
        
        try:
            logger.info("Generating video from existing image...")
            
            video_path = await self.svd_workflow.generate_video(
                image_input=image_input,
                prompt=prompt,
                num_frames=min(video_duration * 1, 2),  # 1 fps, max 2 frames
                num_inference_steps=self.config["video_generation"]["num_inference_steps"],
                guidance_scale=self.config["video_generation"]["guidance_scale"],
                motion_bucket_id=self.config["video_generation"]["motion_bucket_id"],
                noise_aug_strength=self.config["video_generation"]["noise_aug_strength"],
                seed=video_seed
            )
            
            logger.info("✅ Video generated from image successfully")
            return video_path
            
        except Exception as e:
            logger.error(f"Image-to-video generation failed: {e}")
            raise RuntimeError(f"Image-to-video generation failed: {e}")
    
    def unload_models(self) -> bool:
        """Unload all models to free memory"""
        try:
            success = True
            
            if self.text_to_image:
                success &= self.text_to_image.unload_model()
                self.text_to_image = None
            
            if self.svd_workflow:
                success &= self.svd_workflow.unload_model()
                self.svd_workflow = None
            
            self.is_loaded = False
            logger.info("Text-to-video pipeline unloaded")
            return success
            
        except Exception as e:
            logger.error(f"Error unloading pipeline: {e}")
            return False
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            "name": "Text-to-Video Pipeline",
            "description": "Complete text-to-video generation pipeline",
            "loaded": self.is_loaded,
            "device": self.device,
            "dtype": str(self.dtype),
            "components": {
                "text_to_image": self.text_to_image.get_model_info() if self.text_to_image else None,
                "svd_workflow": self.svd_workflow.get_model_info() if self.svd_workflow else None
            },
            "config": self.config,
            "capabilities": [
                "text-to-video",
                "image-to-video", 
                "customizable parameters",
                "seed control",
                "memory efficient"
            ]
        }
    
    def update_config(self, **kwargs):
        """Update pipeline configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        logger.info(f"Updated pipeline config: {kwargs}")
    
    async def generate_image_only(self, prompt: str, **kwargs) -> Image.Image:
        """Generate only an image (useful for preview)"""
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_models() first.")
        
        return await self.text_to_image.generate_image(prompt, **kwargs)
