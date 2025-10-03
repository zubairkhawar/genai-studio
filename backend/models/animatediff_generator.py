"""
AnimateDiff Video Generator - Direct Integration with Existing Stable Diffusion

This module provides AnimateDiff-based video generation using your existing Stable Diffusion
model and AnimateDiff motion adapter weights for efficient text-to-video generation.
"""

import torch
import os
import asyncio
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Union, Callable, List
import logging
from pathlib import Path
import tempfile
import shutil
import cv2
from diffusers import StableDiffusionPipeline, MotionAdapter, AnimateDiffPipeline, DDIMScheduler
import uuid

logger = logging.getLogger(__name__)

class AnimateDiffGenerator:
    """AnimateDiff-based video generation using existing Stable Diffusion model"""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        # Prefer float32 on Apple Silicon (MPS) for stability
        if device == "mps":
            self.dtype = torch.float32
        else:
            self.dtype = dtype
        
        self.is_loaded = False
        self.pipeline = None
        self.motion_adapter = None
        
        # Paths to existing models - use absolute paths from project root
        project_root = Path(__file__).parent.parent.parent
        self.sd_model_dir = project_root / "models" / "image" / "stable-diffusion"
        # Motion adapter lives under the animatediff repo in a subdirectory named 'motion_adapter'
        self.motion_adapter_dir = project_root / "models" / "video" / "animatediff" / "motion_adapter"
        
        # Configuration optimized for different devices with hardware-specific settings
        self.config = self._get_hardware_optimized_config(device)
    
    def _get_hardware_optimized_config(self, device: str) -> Dict[str, Any]:
        """Get hardware-optimized configuration based on detected GPU"""
        import psutil
        
        # Get system memory for better detection
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if device == "mps":
            # Apple Silicon M1/M2/M3 - 16GB RAM optimized settings
            if total_memory_gb >= 16:
                logger.info("Detected Apple Silicon with 16GB+ RAM - using ultra-minimal M1 settings")
                return {
                    "width": 256,  # Ultra-minimal resolution
                    "height": 256,
                    "num_frames": 8,  # Minimal frames
                    "num_inference_steps": 15,  # Minimal steps
                    "guidance_scale": 7.0,  # Basic quality
                    "motion_scale": 1.2,  # Basic motion
                    "fps": 6,
                    "loop": True,
                    "precision": "float32",
                    "attention_slicing": True,
                    "vae_tiling": True,
                    "memory_fraction": 0.0  # Disable memory limit as suggested
                }
            else:
                # Lower memory Apple Silicon
                logger.info("Detected Apple Silicon with <16GB RAM - using ultra-minimal settings")
                return {
                    "width": 256,  # Very low resolution
                    "height": 256,
                    "num_frames": 8,  # Minimal frames
                    "num_inference_steps": 15,  # Minimal steps
                    "guidance_scale": 7.0,
                    "motion_scale": 1.2,
                    "fps": 6,
                    "loop": True,
                    "precision": "float32",
                    "attention_slicing": True,
                    "vae_tiling": True,
                    "memory_fraction": 0.0  # Disable memory limit
                }
        
        elif device == "cuda":
            # Check if this is AMD 7900 XTX or similar high-end GPU
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0).lower()
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    # Detect AMD 7900 XTX or similar high-VRAM GPU
                    if ("radeon" in gpu_name or "rx" in gpu_name) and memory_gb >= 20:
                        logger.info(f"Detected high-end AMD GPU ({gpu_name}, {memory_gb:.1f}GB VRAM) - using 7900 XTX optimized settings")
                        return {
                            "width": 768,  # High resolution for massive VRAM
                            "height": 768,
                            "num_frames": 36,  # 4.5s clip at 8 FPS
                            "num_inference_steps": 38,  # High steps for fine details
                            "guidance_scale": 7.2,  # Slightly lower for higher resolution
                            "motion_scale": 1.6,  # Bold motion with extra frames
                            "fps": 8,
                            "loop": True,
                            "precision": "float16",  # Fast on AMD GPU
                            "attention_slicing": False,  # Keep everything on GPU
                            "vae_tiling": False,
                            "memory_fraction": 1.0
                        }
                    elif memory_gb >= 12:
                        # High-end NVIDIA or other high-VRAM GPU
                        logger.info(f"Detected high-VRAM GPU ({gpu_name}, {memory_gb:.1f}GB VRAM) - using high-quality settings")
                        return {
                            "width": 512,
                            "height": 512,
                            "num_frames": 24,
                            "num_inference_steps": 30,
                            "guidance_scale": 7.5,
                            "motion_scale": 1.5,
                            "fps": 8,
                            "loop": True,
                            "precision": "float16",
                            "attention_slicing": True,
                            "vae_tiling": False,
                            "memory_fraction": 1.0
                        }
            except Exception as e:
                logger.warning(f"Could not detect GPU specifics: {e}")
            
            # Default CUDA settings
            logger.info("Using default CUDA settings")
            return {
                "width": 512,
                "height": 512,
                "num_frames": 16,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "motion_scale": 1.4,
                "fps": 8,
                "loop": True,
                "precision": "float16",
                "attention_slicing": True,
                "vae_tiling": False,
                "memory_fraction": 0.9
            }
        
        else:
            # CPU fallback
            logger.info("Using CPU fallback settings")
            return {
                "width": 256,
                "height": 256,
                "num_frames": 8,
                "num_inference_steps": 15,
                "guidance_scale": 7.0,
                "motion_scale": 1.2,
                "fps": 6,
                "loop": True,
                "precision": "float32",
                "attention_slicing": True,
                "vae_tiling": True,
                "memory_fraction": 0.5
            }
    
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the AnimateDiff model using existing Stable Diffusion and motion adapter"""
        try:
            if self.is_loaded:
                return True
            
            logger.info("Loading AnimateDiff model with existing Stable Diffusion...")
            
            # Verify Stable Diffusion model exists
            if not self._verify_sd_model():
                logger.error("Stable Diffusion model not found locally")
                return False
            
            # Verify motion adapter exists
            if not self._verify_motion_adapter():
                logger.error("Motion adapter not found")
                return False
            
            # Load motion adapter
            logger.info("Loading motion adapter...")
            self.motion_adapter = MotionAdapter.from_pretrained(
                str(self.motion_adapter_dir),
                torch_dtype=self.dtype
            )
            
            # Load Stable Diffusion pipeline
            logger.info("Loading Stable Diffusion pipeline...")
            sd_pipeline = StableDiffusionPipeline.from_pretrained(
                str(self.sd_model_dir),
                torch_dtype=self.dtype,
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Create AnimateDiff pipeline
            logger.info("Creating AnimateDiff pipeline...")
            self.pipeline = AnimateDiffPipeline.from_pretrained(
                str(self.sd_model_dir),
                motion_adapter=self.motion_adapter,
                torch_dtype=self.dtype,
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to device and optimize
            self.pipeline = self.pipeline.to(self.device)
            
            # Use DDIM scheduler with optimized settings for better frame consistency
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config,
                clip_sample=False,
                timestep_spacing="linspace",
                steps_offset=1,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                prediction_type="epsilon"
            )
            
            # Apply hardware-specific optimizations based on detected configuration
            self._apply_hardware_optimizations()
            
            self.is_loaded = True
            logger.info("✅ AnimateDiff model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AnimateDiff model: {e}")
            return False
    
    def _apply_hardware_optimizations(self):
        """Apply hardware-specific optimizations based on detected configuration"""
        try:
            config = self.config
            
            # Set memory fraction for MPS
            if self.device == "mps" and "memory_fraction" in config:
                import os
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = str(config["memory_fraction"])
                logger.info(f"Set MPS memory fraction to {config['memory_fraction']}")
            
            # Apply attention slicing
            if config.get("attention_slicing", True):
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing()
                    logger.info("Enabled attention slicing")
            
            # Apply VAE optimizations
            if config.get("vae_tiling", False):
                if hasattr(self.pipeline, 'enable_vae_tiling'):
                    self.pipeline.enable_vae_tiling()
                    logger.info("Enabled VAE tiling")
            
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
                logger.info("Enabled VAE slicing")
            
            # Apply CPU offload strategies
            if self.device == "mps":
                # MPS-specific optimizations
                if hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                    self.pipeline.enable_sequential_cpu_offload()
                    logger.info("Enabled sequential CPU offload for MPS")
                else:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("Enabled model CPU offload for MPS")
            elif self.device == "cuda":
                # CUDA optimizations - only offload if not high-VRAM GPU
                if not config.get("attention_slicing", True):  # High-VRAM GPU
                    logger.info("High-VRAM GPU detected - keeping models on GPU")
                else:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("Enabled model CPU offload for CUDA")
            
            logger.info(f"Applied hardware optimizations for {self.device}")
            
        except Exception as e:
            logger.warning(f"Could not apply all hardware optimizations: {e}")
    
    def _verify_sd_model(self) -> bool:
        """Verify Stable Diffusion model is available locally"""
        if not self.sd_model_dir.exists():
            return False
        
        # Check for model files
        model_files = list(self.sd_model_dir.rglob("*.safetensors")) + list(self.sd_model_dir.rglob("*.bin"))
        return len(model_files) > 0
    
    def _verify_motion_adapter(self) -> bool:
        """Verify motion adapter is available"""
        if not self.motion_adapter_dir.exists():
            return False
        
        # Check for motion adapter files
        adapter_files = list(self.motion_adapter_dir.rglob("*.safetensors")) + list(self.motion_adapter_dir.rglob("*.bin"))
        return len(adapter_files) > 0
    
    async def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        motion_scale: Optional[float] = None,
        seed: Optional[int] = None,
        output_format: str = "mp4",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> str:
        """Generate video using AnimateDiff pipeline with existing Stable Diffusion model"""
        
        if not self.is_loaded:
            raise RuntimeError("AnimateDiff model not loaded. Call load_model() first.")
        
        try:
            # Set parameters using hardware-optimized defaults
            width = width or self.config["width"]
            height = height or self.config["height"]
            num_frames = num_frames or self.config["num_frames"]
            num_inference_steps = num_inference_steps or self.config["num_inference_steps"]
            guidance_scale = guidance_scale or self.config["guidance_scale"]
            motion_scale = motion_scale or self.config["motion_scale"]
            seed = seed or 42
            
            # Log hardware-optimized settings
            logger.info(f"Using hardware-optimized settings:")
            logger.info(f"  Resolution: {width}x{height}")
            logger.info(f"  Frames: {num_frames} ({num_frames/8:.1f}s at 8 FPS)")
            logger.info(f"  Inference Steps: {num_inference_steps}")
            logger.info(f"  Guidance Scale: {guidance_scale}")
            logger.info(f"  Motion Scale: {motion_scale}")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Precision: {self.config.get('precision', 'auto')}")
            
            # Set default negative prompt for better quality and frame consistency
            if negative_prompt is None:
                negative_prompt = "blurry, low quality, distorted, artifacts, bad anatomy, deformed, ugly, pixelated, low resolution, grainy, noise, compression artifacts, jpeg artifacts, watermark, text, signature, inconsistent motion, flickering, frame drops, temporal artifacts, motion blur, choppy animation, unstable frames, inconsistent lighting, color shifts, frame interpolation errors"
            
            logger.info(f"Generating AnimateDiff video: '{prompt[:50]}...'")
            logger.info(f"Frames: {num_frames}, Size: {width}x{height}")
            
            if progress_callback:
                progress_callback(10, "Preparing AnimateDiff generation...")
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            if progress_callback:
                progress_callback(30, "Generating video frames...")
            
            # Generate video using AnimateDiff pipeline
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Use proper autocast for the device
            if self.device == "mps":
                # For MPS, use CPU autocast or no autocast
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    motion_scale=motion_scale,
                    generator=generator
                )
            else:
                with torch.autocast(self.device):
                    result = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        motion_scale=motion_scale,
                        generator=generator
                    )
            
            if progress_callback:
                progress_callback(80, "Processing video frames...")
            
            # Convert frames to video
            frames = result.frames[0]  # Get the first (and only) video
            
            # Post-process frames for better consistency
            frames = self._post_process_frames(frames)
            
            output_path = await self._save_video_frames(frames, prompt, output_format)
            
            if progress_callback:
                progress_callback(100, "Video generation completed!")
            
            logger.info(f"✅ Generated AnimateDiff video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"AnimateDiff generation failed: {e}")
            raise RuntimeError(f"AnimateDiff generation failed: {e}")
    
    def _post_process_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Post-process frames to improve consistency and quality"""
        try:
            if len(frames) < 3:
                return frames
            
            processed_frames = []
            
            for i, frame in enumerate(frames):
                # Convert to numpy for processing
                img_array = np.array(frame)
                
                # Apply slight sharpening to improve clarity
                if i > 0 and i < len(frames) - 1:
                    # Apply temporal smoothing for middle frames
                    prev_array = np.array(frames[i-1])
                    next_array = np.array(frames[i+1])
                    
                    # Blend with adjacent frames for smoother motion
                    img_array = (img_array * 0.7 + prev_array * 0.15 + next_array * 0.15).astype(np.uint8)
                
                # Convert back to PIL Image
                processed_frame = Image.fromarray(img_array)
                processed_frames.append(processed_frame)
            
            logger.info(f"Post-processed {len(processed_frames)} frames for better consistency")
            return processed_frames
            
        except Exception as e:
            logger.warning(f"Frame post-processing failed, using original frames: {e}")
            return frames
    
    async def _save_video_frames(self, frames: List[Image.Image], prompt: str, output_format: str = "mp4") -> str:
        """Save video frames to file using FFmpeg for better web compatibility"""
        try:
            # Create output directory
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "outputs" / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with appropriate extension
            filename = f"animatediff_{uuid.uuid4().hex[:8]}.{output_format}"
            output_path = output_dir / filename
            
            # For GIF format, use PIL directly for better compatibility
            if output_format.lower() == "gif":
                return await self._save_frames_as_gif(frames, output_path, prompt)
            
            # For other formats, use FFmpeg
            # Create temporary directory for frames
            temp_dir = Path(tempfile.mkdtemp())
            
            try:
                # Save frames as individual images
                frame_paths = []
                for i, frame in enumerate(frames):
                    frame_path = temp_dir / f"frame_{i:04d}.png"
                    frame.save(frame_path, "PNG")
                    frame_paths.append(frame_path)
                
                # Use FFmpeg to create web-compatible video
                import subprocess
                
                # FFmpeg command for web-compatible video with proper metadata
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite output file
                    "-framerate", str(self.config["fps"]),
                    "-i", str(temp_dir / "frame_%04d.png"),
                ]
                
                # Add format-specific options
                if output_format.lower() == "mp4":
                    ffmpeg_cmd.extend([
                        "-c:v", "libx264",  # Use H.264 codec for web compatibility
                        "-pix_fmt", "yuv420p",  # Ensure web compatibility
                        "-crf", "23",  # Good quality
                        "-preset", "medium",  # Balance between speed and compression
                        "-movflags", "+faststart",  # Enable fast start for web streaming
                    ])
                elif output_format.lower() == "webm":
                    ffmpeg_cmd.extend([
                        "-c:v", "libvpx-vp9",  # VP9 codec for WebM
                        "-crf", "23",
                        "-b:v", "0",  # Variable bitrate
                    ])
                else:
                    # Default to MP4 settings
                    ffmpeg_cmd.extend([
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "23",
                    ])
                
                # Add metadata and output
                ffmpeg_cmd.extend([
                    "-metadata", f"title=AnimateDiff Video: {prompt[:50]}",
                    "-metadata", "comment=Generated with AnimateDiff",
                    "-r", str(self.config["fps"]),  # Set output frame rate
                    str(output_path)
                ])
                
                # Run FFmpeg
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                logger.info(f"FFmpeg output: {result.stdout}")
                
                return str(output_path)
                
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise RuntimeError(f"FFmpeg video creation failed: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Failed to save video: {e}")
    
    async def _save_frames_as_gif(self, frames: List[Image.Image], output_path: Path, prompt: str) -> str:
        """Save frames as GIF using PIL for better compatibility"""
        try:
            # Ensure we have frames
            if not frames:
                raise ValueError("No frames to save")
            
            # Calculate duration based on FPS
            duration = int(1000 / self.config["fps"])  # Convert FPS to milliseconds per frame
            
            # Save the first frame
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,  # Infinite loop
                optimize=True,
                comment=f"AnimateDiff Video: {prompt[:50]}"
            )
            
            logger.info(f"✅ GIF saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save GIF: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models (using existing Stable Diffusion)"""
        return ["stable-diffusion-v1-5"]
    
    def unload_model(self) -> bool:
        """Unload the model to free memory"""
        try:
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
            
            if self.motion_adapter:
                del self.motion_adapter
                self.motion_adapter = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            logger.info("AnimateDiff model unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading AnimateDiff model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "AnimateDiff with Stable Diffusion",
            "description": "Text-to-video generation using existing Stable Diffusion model and AnimateDiff motion adapter",
            "loaded": self.is_loaded,
            "device": self.device,
            "dtype": str(self.dtype),
            "config": self.config,
            "available_models": self.get_available_models(),
            "sd_model_dir": str(self.sd_model_dir),
            "motion_adapter_dir": str(self.motion_adapter_dir),
            "capabilities": [
                "text-to-video",
                "MP4 generation",
                "motion control",
                "local model integration",
                "memory efficient"
            ]
        }
    
    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        
        logger.info(f"Updated AnimateDiff config: {kwargs}")