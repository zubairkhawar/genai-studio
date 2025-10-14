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
from diffusers import StableVideoDiffusionPipeline, EulerDiscreteScheduler
import uuid

logger = logging.getLogger(__name__)

class SVDGenerator:
    """Stable Video Diffusion-based video generation from images"""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        # Prefer float32 on Apple Silicon (MPS) for stability
        if device == "mps":
            self.dtype = torch.float32
        else:
            self.dtype = dtype
        
        self.is_loaded = False
        self.pipeline = None
        
        # Paths to SVD model
        project_root = Path(__file__).parent.parent.parent
        self.svd_model_dir = project_root / "models" / "video" / "svd"
        
        # Configuration optimized for different devices
        self.config = self._get_hardware_optimized_config(device)
    
    def _get_hardware_optimized_config(self, device: str) -> Dict[str, Any]:
        """Get hardware-optimized configuration based on detected GPU"""
        import psutil
        
        # Get system memory for better detection
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if device == "mps":
            # Apple Silicon M1/M2/M3 - 16GB RAM optimized settings
            if total_memory_gb >= 16:
                logger.info("Detected Apple Silicon with 16GB+ RAM - using extreme minimal memory SVD settings")
                return {
                    "width": 96,  # Extreme minimal resolution for MPS compatibility
                    "height": 96,
                    "num_frames": 2,  # Absolute minimum frames
                    "num_inference_steps": 8,  # Minimal steps
                    "motion_bucket_id": 127,  # SVD default
                    "noise_aug_strength": 0.02,  # SVD default
                    "fps": 2,
                    "precision": "float32",
                    "attention_slicing": True,
                    "vae_tiling": True,
                    "memory_fraction": 0.0
                }
            else:
                # Lower memory Apple Silicon
                logger.info("Detected Apple Silicon with <16GB RAM - using ultra-minimal settings")
                return {
                    "width": 96,  # Ultra-minimal resolution
                    "height": 96,
                    "num_frames": 3,  # Absolute minimum
                    "num_inference_steps": 8,  # Minimal steps
                    "motion_bucket_id": 127,
                    "noise_aug_strength": 0.02,
                    "fps": 3,
                    "precision": "float32",
                    "attention_slicing": True,
                    "vae_tiling": True,
                    "memory_fraction": 0.0
                }
        
        elif device == "cuda":
            # Comprehensive CUDA GPU detection for Linux and Windows
            try:
                import torch
                import platform
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0).lower()
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    os_name = platform.system().lower()
                    
                    logger.info(f"Detected CUDA GPU: {gpu_name} ({memory_gb:.1f}GB VRAM) on {os_name}")
                    
                    # Detect AMD 7900 XTX or similar high-VRAM GPU
                    if ("radeon" in gpu_name or "rx" in gpu_name) and memory_gb >= 20:
                        logger.info(f"Detected high-end AMD GPU ({gpu_name}, {memory_gb:.1f}GB VRAM) - using optimized settings")
                        return {
                            "width": 512,  # Match SD resolution for better quality
                            "height": 512,
                            "num_frames": 25,  # More frames
                            "num_inference_steps": 25,
                            "motion_bucket_id": 127,
                            "noise_aug_strength": 0.02,
                            "fps": 8,
                            "precision": "float16",
                            "attention_slicing": False,
                            "vae_tiling": False,
                            "memory_fraction": 1.0
                        }
                    elif memory_gb >= 24:
                        # RTX 4090, RTX 6000, A100, etc. - Maximum performance
                        logger.info(f"Detected high-end NVIDIA GPU ({gpu_name}, {memory_gb:.1f}GB VRAM) - using maximum performance settings")
                        return {
                            "width": 512,
                            "height": 512,
                            "num_frames": 25,
                            "num_inference_steps": 25,
                            "motion_bucket_id": 127,
                            "noise_aug_strength": 0.02,
                            "fps": 8,
                            "precision": "float16",
                            "attention_slicing": False,
                            "vae_tiling": False,
                            "memory_fraction": 1.0
                        }
                    elif memory_gb >= 16:
                        # RTX 4080, RTX 3080 Ti, RTX 4070 Ti, etc. - High performance
                        logger.info(f"Detected high-performance NVIDIA GPU ({gpu_name}, {memory_gb:.1f}GB VRAM) - using high-performance settings")
                        return {
                            "width": 512,
                            "height": 512,
                            "num_frames": 20,
                            "num_inference_steps": 20,
                            "motion_bucket_id": 127,
                            "noise_aug_strength": 0.02,
                            "fps": 8,
                            "precision": "float16",
                            "attention_slicing": True,
                            "vae_tiling": False,
                            "memory_fraction": 0.9
                        }
                    elif memory_gb >= 12:
                        # RTX 4070, RTX 3080, RTX 3070 Ti, etc. - Balanced performance
                        logger.info(f"Detected mid-range NVIDIA GPU ({gpu_name}, {memory_gb:.1f}GB VRAM) - using balanced settings")
                        return {
                            "width": 384,
                            "height": 384,
                            "num_frames": 16,
                            "num_inference_steps": 18,
                            "motion_bucket_id": 127,
                            "noise_aug_strength": 0.02,
                            "fps": 6,
                            "precision": "float16",
                            "attention_slicing": True,
                            "vae_tiling": True,
                            "memory_fraction": 0.85
                        }
                    elif memory_gb >= 8:
                        # RTX 3070, RTX 4060 Ti, RTX 2080, etc. - Memory optimized
                        logger.info(f"Detected 8GB NVIDIA GPU ({gpu_name}, {memory_gb:.1f}GB VRAM) - using memory-optimized settings")
                        return {
                            "width": 256,  # Reduced resolution for 8GB GPU
                            "height": 256,
                            "num_frames": 8,  # Fewer frames to save memory
                            "num_inference_steps": 15,  # Reduced steps
                            "motion_bucket_id": 127,
                            "noise_aug_strength": 0.02,
                            "fps": 6,
                            "precision": "float16",
                            "attention_slicing": True,  # Enable aggressive memory optimization
                            "vae_tiling": True,  # Enable VAE tiling
                            "memory_fraction": 0.8  # Use 80% of available memory
                        }
                    elif memory_gb >= 6:
                        # RTX 3060, RTX 2060, GTX 1080 Ti, etc. - Ultra memory optimized
                        logger.info(f"Detected 6GB NVIDIA GPU ({gpu_name}, {memory_gb:.1f}GB VRAM) - using ultra memory-optimized settings")
                        return {
                            "width": 256,
                            "height": 256,
                            "num_frames": 6,  # Very few frames
                            "num_inference_steps": 12,  # Minimal steps
                            "motion_bucket_id": 127,
                            "noise_aug_strength": 0.02,
                            "fps": 4,
                            "precision": "float16",
                            "attention_slicing": True,
                            "vae_tiling": True,
                            "memory_fraction": 0.7
                        }
                    else:
                        # Very low VRAM GPUs - Minimal settings
                        logger.info(f"Detected low-VRAM NVIDIA GPU ({gpu_name}, {memory_gb:.1f}GB VRAM) - using minimal settings")
                        return {
                            "width": 256,
                            "height": 256,
                            "num_frames": 4,  # Absolute minimum
                            "num_inference_steps": 10,  # Minimal steps
                            "motion_bucket_id": 127,
                            "noise_aug_strength": 0.02,
                            "fps": 3,
                            "precision": "float16",
                            "attention_slicing": True,
                            "vae_tiling": True,
                            "memory_fraction": 0.6
                        }
            except Exception as e:
                logger.warning(f"Could not detect GPU specifics: {e}")
            
            # Default CUDA settings
            logger.info("Using default CUDA settings")
            return {
                "width": 384,
                "height": 384,
                "num_frames": 25,
                "num_inference_steps": 25,
                "motion_bucket_id": 127,
                "noise_aug_strength": 0.02,
                "fps": 8,
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
                "num_frames": 16,
                "num_inference_steps": 20,
                "motion_bucket_id": 127,
                "noise_aug_strength": 0.02,
                "fps": 8,
                "precision": "float32",
                "attention_slicing": True,
                "vae_tiling": True,
                "memory_fraction": 0.5
            }
    
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the SVD model"""
        try:
            if self.is_loaded:
                return True
            
            logger.info("Loading Stable Video Diffusion model...")
            
            # Verify model exists
            if not self._verify_svd_model():
                logger.error("SVD model not found locally")
                return False
            
            # Load SVD pipeline
            logger.info("Loading SVD pipeline...")
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                str(self.svd_model_dir),
                torch_dtype=torch.float32 if self.device == "mps" else torch.float16,
                local_files_only=True,
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Use Euler scheduler for better quality
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Apply optimizations
            self._apply_optimizations()
            
            self.is_loaded = True
            logger.info("✅ SVD model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SVD model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _apply_optimizations(self):
        """Apply device-specific optimizations"""
        try:
            config = self.config
            
            # Set memory fraction for MPS
            if self.device == "mps" and "memory_fraction" in config:
                import os
                # Disable MPS memory limit to allow SVD to use more memory
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = "0.0"
                logger.info("Disabled MPS memory limit for SVD compatibility")
            
            # Apply attention slicing
            if config.get("attention_slicing", True):
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing(1)  # Most aggressive slicing
                    logger.info("Enabled aggressive attention slicing")
            
            # Additional memory optimizations for CUDA with limited VRAM
            if self.device == "cuda":
                import torch
                import platform
                if torch.cuda.is_available():
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    os_name = platform.system().lower()
                    
                    logger.info(f"Applying CUDA optimizations for {memory_gb:.1f}GB GPU on {os_name}")
                    
                    if memory_gb < 12:  # For 8GB and lower GPUs
                        # Enable CPU offload for memory-constrained GPUs
                        if hasattr(self.pipeline, 'enable_cpu_offload'):
                            self.pipeline.enable_cpu_offload()
                            logger.info("Enabled CPU offload for memory-constrained GPU")
                        
                        # Enable sequential CPU offload for very low VRAM
                        if memory_gb < 8 and hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                            self.pipeline.enable_sequential_cpu_offload()
                            logger.info("Enabled sequential CPU offload for very low VRAM")
                        
                        # Clear CUDA cache before processing
                        torch.cuda.empty_cache()
                        logger.info("Cleared CUDA cache for memory optimization")
                    
                    # Platform-specific optimizations
                    if os_name == "windows":
                        # Windows-specific memory management
                        torch.cuda.set_per_process_memory_fraction(0.9)
                        logger.info("Set Windows CUDA memory fraction to 90%")
                    elif os_name == "linux":
                        # Linux-specific optimizations
                        torch.cuda.empty_cache()
                        logger.info("Applied Linux CUDA optimizations")
            
            # Apply VAE optimizations
            if config.get("vae_tiling", False):
                if hasattr(self.pipeline, 'enable_vae_tiling'):
                    self.pipeline.enable_vae_tiling()
                    logger.info("Enabled VAE tiling")
            
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
                logger.info("Enabled VAE slicing")
            
            # Additional memory optimizations for MPS
            if self.device == "mps":
                if hasattr(self.pipeline, 'enable_cpu_offload'):
                    self.pipeline.enable_cpu_offload()
                    logger.info("Enabled CPU offload for MPS")
                
                # Force minimal memory usage
                import torch
                if hasattr(torch, 'mps'):
                    torch.mps.empty_cache()
                    logger.info("Cleared MPS cache")

            # Force upcast VAE to float32 on MPS to avoid brownish/washed colors
            try:
                if self.device == "mps" and hasattr(self.pipeline, 'vae'):
                    self.pipeline.vae.to(torch.float32)
                    if hasattr(self.pipeline.vae.config, 'force_upcast'):
                        self.pipeline.vae.config.force_upcast = True
                    logger.info("VAE upcast to float32 for MPS")
            except Exception as e:
                logger.warning(f"Could not upcast VAE: {e}")
            
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
            
            logger.info(f"Applied optimizations for {self.device}")
            
        except Exception as e:
            logger.warning(f"Could not apply all optimizations: {e}")
    
    def _verify_svd_model(self) -> bool:
        """Verify SVD model is available locally"""
        if not self.svd_model_dir.exists():
            return False
        
        # Check for essential files only
        essential_files = [
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors"
        ]
        
        for file_path in essential_files:
            if not (self.svd_model_dir / file_path).exists():
                logger.error(f"Missing essential SVD file: {self.svd_model_dir / file_path}")
                return False
        
        # Check for any model weight files
        weight_files = list(self.svd_model_dir.rglob("*.safetensors")) + list(self.svd_model_dir.rglob("*.bin"))
        if len(weight_files) == 0:
            logger.error("No SVD model weight files found")
            return False
        
        logger.info(f"SVD model verification passed: {len(weight_files)} weight files found")
        return True
    
    async def generate_video(
        self,
        init_image: Image.Image,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        motion_bucket_id: Optional[int] = None,
        noise_aug_strength: Optional[float] = None,
        seed: Optional[int] = None,
        output_format: str = "mp4",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> str:
        """Generate video using SVD pipeline from an input image"""
        
        if not self.is_loaded:
            raise RuntimeError("SVD model not loaded. Call load_model() first.")
        
        try:
            # Use config defaults
            width = width or self.config["width"]
            height = height or self.config["height"]
            num_frames = num_frames or self.config["num_frames"]
            num_inference_steps = num_inference_steps or self.config["num_inference_steps"]
            motion_bucket_id = motion_bucket_id or self.config["motion_bucket_id"]
            noise_aug_strength = noise_aug_strength or self.config["noise_aug_strength"]
            seed = seed or 42
            
            logger.info(f"Generating SVD video: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")
            
            if progress_callback:
                progress_callback(10, "Preparing SVD generation...")
            
            # Set seed
            torch.manual_seed(seed)
            
            if progress_callback:
                progress_callback(30, "Generating video frames...")
            
            # Prepare generator
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Resize input image to match SVD requirements
            if init_image.size != (width, height):
                logger.info(f"Resizing input image from {init_image.size} to {width}x{height}")
                init_image = init_image.resize((width, height), Image.BICUBIC)
            
            # Generate with SVD
            logger.info("Starting SVD generation...")
            
            # Clear memory before generation to prevent buffer overflow
            if self.device == "cuda":
                torch.cuda.empty_cache()
                # Additional memory management for different platforms
                import platform
                os_name = platform.system().lower()
                if os_name == "windows":
                    # Windows-specific memory management
                    torch.cuda.synchronize()
                    logger.info("Synchronized CUDA on Windows before SVD generation")
                elif os_name == "linux":
                    # Linux-specific memory management
                    torch.cuda.empty_cache()
                    logger.info("Applied Linux CUDA memory management before SVD generation")
                else:
                    logger.info("Cleared CUDA cache before SVD generation")
            elif self.device == "mps":
                if hasattr(torch, 'mps'):
                    torch.mps.empty_cache()
                    logger.info("Cleared MPS cache before SVD generation")
            
            with torch.no_grad():
                result = self.pipeline(
                    image=init_image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    motion_bucket_id=motion_bucket_id,
                    noise_aug_strength=noise_aug_strength,
                    generator=generator,
                    decode_chunk_size=4,  # Smaller chunks for memory-constrained GPUs
                )
            
            if progress_callback:
                progress_callback(80, "Processing video frames...")
            
            # Get frames
            frames = result.frames[0] if hasattr(result, 'frames') else result.images
            
            # Save video
            output_path = await self._save_video_frames(frames, output_format)
            
            # Clear memory after generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                # Platform-specific cleanup
                import platform
                os_name = platform.system().lower()
                if os_name == "windows":
                    torch.cuda.synchronize()
                    logger.info("Synchronized CUDA on Windows after SVD generation")
                elif os_name == "linux":
                    torch.cuda.empty_cache()
                    logger.info("Applied Linux CUDA cleanup after SVD generation")
                else:
                    logger.info("Cleared CUDA cache after SVD generation")
            elif self.device == "mps":
                if hasattr(torch, 'mps'):
                    torch.mps.empty_cache()
                    logger.info("Cleared MPS cache after SVD generation")
            
            logger.info(f"✅ Generated SVD video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"SVD generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"SVD generation failed: {e}")
    
    async def _save_video_frames(self, frames: List[Image.Image], output_format: str = "mp4") -> str:
        """Save video frames to file using FFmpeg"""
        try:
            # Create output directory
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "outputs" / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with appropriate extension
            filename = f"svd_{uuid.uuid4().hex[:8]}.{output_format}"
            output_path = output_dir / filename
            
            # For GIF format, use PIL directly
            if output_format.lower() == "gif":
                return await self._save_frames_as_gif(frames, output_path)
            
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
                
                # Use FFmpeg to create video
                import subprocess
                
                # FFmpeg command for web-compatible video
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
                    "-metadata", "title=SVD Video",
                    "-metadata", "comment=Generated with Stable Video Diffusion",
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
    
    async def _save_frames_as_gif(self, frames: List[Image.Image], output_path: Path) -> str:
        """Save frames as GIF using PIL"""
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
                comment="SVD Video"
            )
            
            logger.info(f"✅ GIF saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save GIF: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return ["stable-video-diffusion-img2vid-xt"]
    
    def unload_model(self) -> bool:
        """Unload the model to free memory"""
        try:
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
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
            "description": "Image-to-video generation using Stable Video Diffusion",
            "loaded": self.is_loaded,
            "device": self.device,
            "dtype": str(self.dtype),
            "config": self.config,
            "available_models": self.get_available_models(),
            "svd_model_dir": str(self.svd_model_dir),
            "capabilities": [
                "image-to-video",
                "MP4 generation",
                "motion generation",
                "local model integration",
                "memory efficient"
            ]
        }
    
    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        
        logger.info(f"Updated SVD config: {kwargs}")
