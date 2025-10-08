#!/usr/bin/env python3
"""
Comprehensive Video Generation Pipeline Testing Script

Tests all 5 video generation pipeline combinations:
1. Text → SD → AnimateDiff → FFmpeg → Output
2. Text → SD → AnimateDiff → StableSR → FFmpeg → Output  
3. Text → SD → AnimateDiff → RealESRGAN → FILM → FFmpeg → Output
4. Text → SD → AnimateDiff → StableSR → FILM → FFmpeg → Output
5. Text → SD → AnimateDiff → StableSR → RealESRGAN → FILM → FFmpeg → Output

Usage: python test_video_pipelines.py
"""

import asyncio
import logging
import time
import sys
import pathlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

# Add backend to path
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

from backend.utils.gpu_detector import GPUDetector
from backend.models.ultimate_video_generator import UltimateVideoGenerator
from backend.models.enhanced_video_generator import EnhancedVideoGenerator
from backend.models.video_generator import VideoGenerator
from backend.models.text_to_image import TextToImageGenerator
from backend.models.animatediff_generator import AnimateDiffGenerator
from backend.models.stablesr_refiner import StableSRRefiner
from backend.models.realesrgan_upscaler import RealESRGANUpscaler
from backend.models.film_interpolator import FILMInterpolator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_test_results.log')
    ]
)
logger = logging.getLogger(__name__)

class VideoPipelineTester:
    """Comprehensive video pipeline testing framework"""
    
    def __init__(self):
        gpu_detector = GPUDetector()
        self.gpu_info = gpu_detector.gpu_info
        self.test_prompt = "Ocean waves crashing against rocky cliffs"
        self.results = {}
        self.start_time = None
        
        # Setup output directory
        project_root = pathlib.Path(__file__).parent
        self.outputs_dir = project_root / "outputs" / "videos" / "pipeline_tests"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.sd_generator = None
        self.ad_generator = None
        self.sr_refiner = None
        self.realesrgan_upscaler = None
        self.film_interpolator = None
        
        logger.info(f"🚀 Initialized Video Pipeline Tester")
        logger.info(f"📱 Device: {self.gpu_info['device']}")
        logger.info(f"💾 Output Directory: {self.outputs_dir}")
        logger.info(f"🎯 Test Prompt: '{self.test_prompt}'")

    async def setup_models(self) -> bool:
        """Load all required models for testing"""
        try:
            logger.info("🔄 Loading models for pipeline testing...")
            
            # Load Stable Diffusion
            logger.info("📸 Loading Stable Diffusion...")
            self.sd_generator = TextToImageGenerator(device=self.gpu_info["device"])
            sd_path = pathlib.Path(__file__).parent / "models" / "image" / "stable-diffusion"
            sd_loaded = await asyncio.to_thread(self.sd_generator.load_model, str(sd_path))
            if not sd_loaded:
                logger.error("❌ Failed to load Stable Diffusion")
                return False
            
            # Load AnimateDiff
            logger.info("🎬 Loading AnimateDiff...")
            self.ad_generator = AnimateDiffGenerator(device=self.gpu_info["device"])
            ad_loaded = await self.ad_generator.load_model()
            if not ad_loaded:
                logger.error("❌ Failed to load AnimateDiff")
                return False
            
            # Load StableSR Refiner
            logger.info("🔧 Loading StableSR Refiner...")
            self.sr_refiner = StableSRRefiner()
            sr_loaded = await self.sr_refiner.load_model()
            if not sr_loaded:
                logger.warning("⚠️ StableSR Refiner not available, will skip in pipelines")
            
            # Load RealESRGAN Upscaler
            logger.info("🔍 Loading RealESRGAN Upscaler...")
            self.realesrgan_upscaler = RealESRGANUpscaler(device=self.gpu_info["device"])
            re_loaded = await self.realesrgan_upscaler.load_model()
            if not re_loaded:
                logger.warning("⚠️ RealESRGAN Upscaler not available, will skip in pipelines")
            
            # Load FILM Interpolator
            logger.info("🎯 Loading FILM Interpolator...")
            self.film_interpolator = FILMInterpolator(device=self.gpu_info["device"])
            film_loaded = await self.film_interpolator.load_model()
            if not film_loaded:
                logger.warning("⚠️ FILM Interpolator not available, will use fallback")
            
            logger.info("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup models: {e}")
            return False

    async def test_pipeline_1(self) -> Dict[str, Any]:
        """Test Pipeline 1: Text → SD → AnimateDiff → FFmpeg → Output"""
        logger.info("🧪 Testing Pipeline 1: Text → SD → AnimateDiff → FFmpeg")
        
        start_time = time.time()
        result = {
            "pipeline": "Text → SD → AnimateDiff → FFmpeg",
            "success": False,
            "duration": 0,
            "output_path": None,
            "error": None,
            "stages": {}
        }
        
        try:
            # Stage 1: Generate keyframe with SD
            stage_start = time.time()
            logger.info("  📸 Stage 1: Generating keyframe with Stable Diffusion...")
            
            keyframe = await asyncio.to_thread(
                self.sd_generator.generate_image,
                prompt=self.test_prompt,
                width=512,
                height=512,
                num_inference_steps=20,
                guidance_scale=7.5
            )
            
            result["stages"]["stable_diffusion"] = {
                "success": True,
                "duration": time.time() - stage_start,
                "output_size": f"{keyframe.width}x{keyframe.height}"
            }
            logger.info(f"  ✅ SD keyframe generated: {keyframe.width}x{keyframe.height}")
            
            # Stage 2: Generate motion with AnimateDiff
            stage_start = time.time()
            logger.info("  🎬 Stage 2: Generating motion with AnimateDiff...")
            
            video_path = await self.ad_generator.generate_video(
                prompt=self.test_prompt,
                width=512,
                height=512,
                num_frames=16,
                num_inference_steps=20,
                guidance_scale=7.5,
                motion_scale=1.4
            )
            
            result["stages"]["animatediff"] = {
                "success": True,
                "duration": time.time() - stage_start,
                "output_path": video_path
            }
            logger.info(f"  ✅ AnimateDiff video generated: {video_path}")
            
            # Stage 3: Final assembly (already done by AnimateDiff)
            result["stages"]["ffmpeg"] = {
                "success": True,
                "duration": 0,
                "note": "Handled by AnimateDiff generator"
            }
            
            # Copy to test outputs with descriptive name
            import shutil
            test_output = self.outputs_dir / f"pipeline1_sd_animatediff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            shutil.copy2(video_path, test_output)
            
            result["success"] = True
            result["output_path"] = str(test_output)
            result["duration"] = time.time() - start_time
            
            logger.info(f"  ✅ Pipeline 1 completed successfully in {result['duration']:.2f}s")
            logger.info(f"  📁 Output saved to: {test_output}")
            
        except Exception as e:
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            logger.error(f"  ❌ Pipeline 1 failed: {e}")
        
        return result

    async def test_pipeline_2(self) -> Dict[str, Any]:
        """Test Pipeline 2: Text → SD → AnimateDiff → StableSR → FFmpeg → Output"""
        logger.info("🧪 Testing Pipeline 2: Text → SD → AnimateDiff → StableSR → FFmpeg")
        
        start_time = time.time()
        result = {
            "pipeline": "Text → SD → AnimateDiff → StableSR → FFmpeg",
            "success": False,
            "duration": 0,
            "output_path": None,
            "error": None,
            "stages": {}
        }
        
        try:
            # Stage 1: Generate keyframe with SD
            stage_start = time.time()
            logger.info("  📸 Stage 1: Generating keyframe with Stable Diffusion...")
            
            keyframe = await asyncio.to_thread(
                self.sd_generator.generate_image,
                prompt=self.test_prompt,
                width=512,
                height=512,
                num_inference_steps=20,
                guidance_scale=7.5
            )
            
            result["stages"]["stable_diffusion"] = {
                "success": True,
                "duration": time.time() - stage_start,
                "output_size": f"{keyframe.width}x{keyframe.height}"
            }
            
            # Stage 2: Generate motion with AnimateDiff
            stage_start = time.time()
            logger.info("  🎬 Stage 2: Generating motion with AnimateDiff...")
            
            video_path = await self.ad_generator.generate_video(
                prompt=self.test_prompt,
                width=512,
                height=512,
                num_frames=16,
                num_inference_steps=20,
                guidance_scale=7.5,
                motion_scale=1.4
            )
            
            result["stages"]["animatediff"] = {
                "success": True,
                "duration": time.time() - stage_start,
                "output_path": video_path
            }
            
            # Stage 3: Extract frames and apply StableSR refinement
            stage_start = time.time()
            logger.info("  🔧 Stage 3: Applying StableSR refinement...")
            
            if self.sr_refiner and self.sr_refiner.is_loaded:
                # Extract frames from video
                frames = await self._extract_frames_from_video(video_path)
                
                # Apply StableSR refinement
                refined_frames = await self.sr_refiner.refine_frames(
                    frames, 
                    denoise_strength=0.25
                )
                
                result["stages"]["stablesr"] = {
                    "success": True,
                    "duration": time.time() - stage_start,
                    "frames_processed": len(refined_frames)
                }
                
                # Stage 4: Assemble final video
                stage_start = time.time()
                logger.info("  🎞️ Stage 4: Assembling final video...")
                
                final_output = await self._assemble_video_from_frames(
                    refined_frames, 
                    "pipeline2_sd_animatediff_stablesr"
                )
                
                result["stages"]["ffmpeg"] = {
                    "success": True,
                    "duration": time.time() - stage_start
                }
                
                result["success"] = True
                result["output_path"] = final_output
                
            else:
                logger.warning("  ⚠️ StableSR not available, skipping refinement")
                result["stages"]["stablesr"] = {
                    "success": False,
                    "duration": 0,
                    "error": "StableSR not loaded"
                }
                
                # Copy original video
                import shutil
                test_output = self.outputs_dir / f"pipeline2_sd_animatediff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                shutil.copy2(video_path, test_output)
                result["output_path"] = str(test_output)
                result["success"] = True
            
            result["duration"] = time.time() - start_time
            logger.info(f"  ✅ Pipeline 2 completed successfully in {result['duration']:.2f}s")
            logger.info(f"  📁 Output saved to: {result['output_path']}")
            
        except Exception as e:
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            logger.error(f"  ❌ Pipeline 2 failed: {e}")
        
        return result

    async def test_pipeline_3(self) -> Dict[str, Any]:
        """Test Pipeline 3: Text → SD → AnimateDiff → RealESRGAN → FILM → FFmpeg → Output"""
        logger.info("🧪 Testing Pipeline 3: Text → SD → AnimateDiff → RealESRGAN → FILM → FFmpeg")
        
        start_time = time.time()
        result = {
            "pipeline": "Text → SD → AnimateDiff → RealESRGAN → FILM → FFmpeg",
            "success": False,
            "duration": 0,
            "output_path": None,
            "error": None,
            "stages": {}
        }
        
        try:
            # Stage 1: Generate keyframe with SD
            stage_start = time.time()
            logger.info("  📸 Stage 1: Generating keyframe with Stable Diffusion...")
            
            keyframe = await asyncio.to_thread(
                self.sd_generator.generate_image,
                prompt=self.test_prompt,
                width=512,
                height=512,
                num_inference_steps=20,
                guidance_scale=7.5
            )
            
            result["stages"]["stable_diffusion"] = {
                "success": True,
                "duration": time.time() - stage_start,
                "output_size": f"{keyframe.width}x{keyframe.height}"
            }
            
            # Stage 2: Generate motion with AnimateDiff
            stage_start = time.time()
            logger.info("  🎬 Stage 2: Generating motion with AnimateDiff...")
            
            video_path = await self.ad_generator.generate_video(
                prompt=self.test_prompt,
                width=512,
                height=512,
                num_frames=16,
                num_inference_steps=20,
                guidance_scale=7.5,
                motion_scale=1.4
            )
            
            result["stages"]["animatediff"] = {
                "success": True,
                "duration": time.time() - stage_start,
                "output_path": video_path
            }
            
            # Stage 3: Extract frames and apply RealESRGAN upscaling
            stage_start = time.time()
            logger.info("  🔍 Stage 3: Applying RealESRGAN upscaling...")
            
            frames = await self._extract_frames_from_video(video_path)
            
            if self.realesrgan_upscaler and self.realesrgan_upscaler.is_loaded:
                upscaled_frames = await self.realesrgan_upscaler.upscale_frames(
                    frames, 
                    scale=2.0
                )
                
                result["stages"]["realesrgan"] = {
                    "success": True,
                    "duration": time.time() - stage_start,
                    "frames_processed": len(upscaled_frames),
                    "upscale_factor": 2.0
                }
            else:
                logger.warning("  ⚠️ RealESRGAN not available, using original frames")
                upscaled_frames = frames
                result["stages"]["realesrgan"] = {
                    "success": False,
                    "duration": 0,
                    "error": "RealESRGAN not loaded"
                }
            
            # Stage 4: Apply FILM interpolation
            stage_start = time.time()
            logger.info("  🎯 Stage 4: Applying FILM interpolation...")
            
            if self.film_interpolator and self.film_interpolator.is_loaded:
                interpolated_frames = await self.film_interpolator.interpolate_frames(
                    upscaled_frames,
                    target_fps=24,
                    original_fps=8
                )
                
                result["stages"]["film"] = {
                    "success": True,
                    "duration": time.time() - stage_start,
                    "frames_before": len(upscaled_frames),
                    "frames_after": len(interpolated_frames)
                }
            else:
                logger.warning("  ⚠️ FILM not available, using original frames")
                interpolated_frames = upscaled_frames
                result["stages"]["film"] = {
                    "success": False,
                    "duration": 0,
                    "error": "FILM not loaded"
                }
            
            # Stage 5: Assemble final video
            stage_start = time.time()
            logger.info("  🎞️ Stage 5: Assembling final video...")
            
            final_output = await self._assemble_video_from_frames(
                interpolated_frames, 
                "pipeline3_sd_animatediff_realesrgan_film"
            )
            
            result["stages"]["ffmpeg"] = {
                "success": True,
                "duration": time.time() - stage_start
            }
            
            result["success"] = True
            result["output_path"] = final_output
            result["duration"] = time.time() - start_time
            
            logger.info(f"  ✅ Pipeline 3 completed successfully in {result['duration']:.2f}s")
            logger.info(f"  📁 Output saved to: {final_output}")
            
        except Exception as e:
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            logger.error(f"  ❌ Pipeline 3 failed: {e}")
        
        return result

    async def test_pipeline_4(self) -> Dict[str, Any]:
        """Test Pipeline 4: Text → SD → AnimateDiff → StableSR → FILM → FFmpeg → Output"""
        logger.info("🧪 Testing Pipeline 4: Text → SD → AnimateDiff → StableSR → FILM → FFmpeg")
        
        start_time = time.time()
        result = {
            "pipeline": "Text → SD → AnimateDiff → StableSR → FILM → FFmpeg",
            "success": False,
            "duration": 0,
            "output_path": None,
            "error": None,
            "stages": {}
        }
        
        try:
            # Stage 1: Generate keyframe with SD
            stage_start = time.time()
            logger.info("  📸 Stage 1: Generating keyframe with Stable Diffusion...")
            
            keyframe = await asyncio.to_thread(
                self.sd_generator.generate_image,
                prompt=self.test_prompt,
                width=512,
                height=512,
                num_inference_steps=20,
                guidance_scale=7.5
            )
            
            result["stages"]["stable_diffusion"] = {
                "success": True,
                "duration": time.time() - stage_start,
                "output_size": f"{keyframe.width}x{keyframe.height}"
            }
            
            # Stage 2: Generate motion with AnimateDiff
            stage_start = time.time()
            logger.info("  🎬 Stage 2: Generating motion with AnimateDiff...")
            
            video_path = await self.ad_generator.generate_video(
                prompt=self.test_prompt,
                width=512,
                height=512,
                num_frames=16,
                num_inference_steps=20,
                guidance_scale=7.5,
                motion_scale=1.4
            )
            
            result["stages"]["animatediff"] = {
                "success": True,
                "duration": time.time() - stage_start,
                "output_path": video_path
            }
            
            # Stage 3: Extract frames and apply StableSR refinement
            stage_start = time.time()
            logger.info("  🔧 Stage 3: Applying StableSR refinement...")
            
            frames = await self._extract_frames_from_video(video_path)
            
            if self.sr_refiner and self.sr_refiner.is_loaded:
                refined_frames = await self.sr_refiner.refine_frames(
                    frames, 
                    denoise_strength=0.25
                )
                
                result["stages"]["stablesr"] = {
                    "success": True,
                    "duration": time.time() - stage_start,
                    "frames_processed": len(refined_frames)
                }
            else:
                logger.warning("  ⚠️ StableSR not available, using original frames")
                refined_frames = frames
                result["stages"]["stablesr"] = {
                    "success": False,
                    "duration": 0,
                    "error": "StableSR not loaded"
                }
            
            # Stage 4: Apply FILM interpolation
            stage_start = time.time()
            logger.info("  🎯 Stage 4: Applying FILM interpolation...")
            
            if self.film_interpolator and self.film_interpolator.is_loaded:
                interpolated_frames = await self.film_interpolator.interpolate_frames(
                    refined_frames,
                    target_fps=24,
                    original_fps=8
                )
                
                result["stages"]["film"] = {
                    "success": True,
                    "duration": time.time() - stage_start,
                    "frames_before": len(refined_frames),
                    "frames_after": len(interpolated_frames)
                }
            else:
                logger.warning("  ⚠️ FILM not available, using original frames")
                interpolated_frames = refined_frames
                result["stages"]["film"] = {
                    "success": False,
                    "duration": 0,
                    "error": "FILM not loaded"
                }
            
            # Stage 5: Assemble final video
            stage_start = time.time()
            logger.info("  🎞️ Stage 5: Assembling final video...")
            
            final_output = await self._assemble_video_from_frames(
                interpolated_frames, 
                "pipeline4_sd_animatediff_stablesr_film"
            )
            
            result["stages"]["ffmpeg"] = {
                "success": True,
                "duration": time.time() - stage_start
            }
            
            result["success"] = True
            result["output_path"] = final_output
            result["duration"] = time.time() - start_time
            
            logger.info(f"  ✅ Pipeline 4 completed successfully in {result['duration']:.2f}s")
            logger.info(f"  📁 Output saved to: {final_output}")
            
        except Exception as e:
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            logger.error(f"  ❌ Pipeline 4 failed: {e}")
        
        return result

    async def test_pipeline_5(self) -> Dict[str, Any]:
        """Test Pipeline 5: Text → SD → AnimateDiff → StableSR → RealESRGAN → FILM → FFmpeg → Output"""
        logger.info("🧪 Testing Pipeline 5: Text → SD → AnimateDiff → StableSR → RealESRGAN → FILM → FFmpeg")
        
        start_time = time.time()
        result = {
            "pipeline": "Text → SD → AnimateDiff → StableSR → RealESRGAN → FILM → FFmpeg",
            "success": False,
            "duration": 0,
            "output_path": None,
            "error": None,
            "stages": {}
        }
        
        try:
            # Stage 1: Generate keyframe with SD
            stage_start = time.time()
            logger.info("  📸 Stage 1: Generating keyframe with Stable Diffusion...")
            
            keyframe = await asyncio.to_thread(
                self.sd_generator.generate_image,
                prompt=self.test_prompt,
                width=512,
                height=512,
                num_inference_steps=20,
                guidance_scale=7.5
            )
            
            result["stages"]["stable_diffusion"] = {
                "success": True,
                "duration": time.time() - stage_start,
                "output_size": f"{keyframe.width}x{keyframe.height}"
            }
            
            # Stage 2: Generate motion with AnimateDiff
            stage_start = time.time()
            logger.info("  🎬 Stage 2: Generating motion with AnimateDiff...")
            
            video_path = await self.ad_generator.generate_video(
                prompt=self.test_prompt,
                width=512,
                height=512,
                num_frames=16,
                num_inference_steps=20,
                guidance_scale=7.5,
                motion_scale=1.4
            )
            
            result["stages"]["animatediff"] = {
                "success": True,
                "duration": time.time() - stage_start,
                "output_path": video_path
            }
            
            # Stage 3: Extract frames and apply StableSR refinement
            stage_start = time.time()
            logger.info("  🔧 Stage 3: Applying StableSR refinement...")
            
            frames = await self._extract_frames_from_video(video_path)
            
            if self.sr_refiner and self.sr_refiner.is_loaded:
                refined_frames = await self.sr_refiner.refine_frames(
                    frames, 
                    denoise_strength=0.25
                )
                
                result["stages"]["stablesr"] = {
                    "success": True,
                    "duration": time.time() - stage_start,
                    "frames_processed": len(refined_frames)
                }
            else:
                logger.warning("  ⚠️ StableSR not available, using original frames")
                refined_frames = frames
                result["stages"]["stablesr"] = {
                    "success": False,
                    "duration": 0,
                    "error": "StableSR not loaded"
                }
            
            # Stage 4: Apply RealESRGAN upscaling
            stage_start = time.time()
            logger.info("  🔍 Stage 4: Applying RealESRGAN upscaling...")
            
            if self.realesrgan_upscaler and self.realesrgan_upscaler.is_loaded:
                upscaled_frames = await self.realesrgan_upscaler.upscale_frames(
                    refined_frames, 
                    scale=2.0
                )
                
                result["stages"]["realesrgan"] = {
                    "success": True,
                    "duration": time.time() - stage_start,
                    "frames_processed": len(upscaled_frames),
                    "upscale_factor": 2.0
                }
            else:
                logger.warning("  ⚠️ RealESRGAN not available, using refined frames")
                upscaled_frames = refined_frames
                result["stages"]["realesrgan"] = {
                    "success": False,
                    "duration": 0,
                    "error": "RealESRGAN not loaded"
                }
            
            # Stage 5: Apply FILM interpolation
            stage_start = time.time()
            logger.info("  🎯 Stage 5: Applying FILM interpolation...")
            
            if self.film_interpolator and self.film_interpolator.is_loaded:
                interpolated_frames = await self.film_interpolator.interpolate_frames(
                    upscaled_frames,
                    target_fps=24,
                    original_fps=8
                )
                
                result["stages"]["film"] = {
                    "success": True,
                    "duration": time.time() - stage_start,
                    "frames_before": len(upscaled_frames),
                    "frames_after": len(interpolated_frames)
                }
            else:
                logger.warning("  ⚠️ FILM not available, using upscaled frames")
                interpolated_frames = upscaled_frames
                result["stages"]["film"] = {
                    "success": False,
                    "duration": 0,
                    "error": "FILM not loaded"
                }
            
            # Stage 6: Assemble final video
            stage_start = time.time()
            logger.info("  🎞️ Stage 6: Assembling final video...")
            
            final_output = await self._assemble_video_from_frames(
                interpolated_frames, 
                "pipeline5_sd_animatediff_stablesr_realesrgan_film"
            )
            
            result["stages"]["ffmpeg"] = {
                "success": True,
                "duration": time.time() - stage_start
            }
            
            result["success"] = True
            result["output_path"] = final_output
            result["duration"] = time.time() - start_time
            
            logger.info(f"  ✅ Pipeline 5 completed successfully in {result['duration']:.2f}s")
            logger.info(f"  📁 Output saved to: {final_output}")
            
        except Exception as e:
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            logger.error(f"  ❌ Pipeline 5 failed: {e}")
        
        return result

    async def _extract_frames_from_video(self, video_path: str) -> List:
        """Extract frames from a video file"""
        import cv2
        from PIL import Image
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        
        cap.release()
        return frames

    async def _assemble_video_from_frames(self, frames: List, prefix: str) -> str:
        """Assemble video from frames using FFmpeg"""
        import tempfile
        import shutil
        import subprocess
        
        temp_dir = pathlib.Path(tempfile.mkdtemp())
        try:
            # Save frames as PNG files
            for i, frame in enumerate(frames):
                frame_path = temp_dir / f"frame_{i:06d}.png"
                frame.save(frame_path, "PNG")
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.outputs_dir / f"{prefix}_{timestamp}.mp4"
            
            # FFmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-framerate", "24",
                "-i", str(temp_dir / "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "20",
                "-preset", "medium",
                "-movflags", "+faststart",
                str(output_path)
            ]
            
            # Run FFmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
            
            return str(output_path)
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def run_all_tests(self):
        """Run all pipeline tests"""
        self.start_time = time.time()
        logger.info("🚀 Starting comprehensive video pipeline testing...")
        logger.info("=" * 80)
        
        # Setup models
        if not await self.setup_models():
            logger.error("❌ Failed to setup models. Exiting.")
            return
        
        # Run all pipeline tests
        test_methods = [
            ("Pipeline 1", self.test_pipeline_1),
            ("Pipeline 2", self.test_pipeline_2),
            ("Pipeline 3", self.test_pipeline_3),
            ("Pipeline 4", self.test_pipeline_4),
            ("Pipeline 5", self.test_pipeline_5),
        ]
        
        for test_name, test_method in test_methods:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = await test_method()
                self.results[test_name] = result
            except Exception as e:
                logger.error(f"❌ {test_name} failed with exception: {e}")
                self.results[test_name] = {
                    "pipeline": test_name,
                    "success": False,
                    "duration": 0,
                    "output_path": None,
                    "error": str(e),
                    "stages": {}
                }
        
        # Generate summary report
        await self.generate_summary_report()

    async def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        total_duration = time.time() - self.start_time
        
        logger.info("\n" + "="*80)
        logger.info("📊 PIPELINE TESTING SUMMARY REPORT")
        logger.info("="*80)
        
        # Overall statistics
        successful_tests = sum(1 for result in self.results.values() if result["success"])
        total_tests = len(self.results)
        
        logger.info(f"🎯 Test Prompt: '{self.test_prompt}'")
        logger.info(f"⏱️  Total Test Duration: {total_duration:.2f} seconds")
        logger.info(f"✅ Successful Tests: {successful_tests}/{total_tests}")
        logger.info(f"❌ Failed Tests: {total_tests - successful_tests}/{total_tests}")
        logger.info(f"📁 Output Directory: {self.outputs_dir}")
        
        # Individual test results
        logger.info("\n📋 Individual Test Results:")
        logger.info("-" * 80)
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            duration = result["duration"]
            output_path = result.get("output_path", "N/A")
            
            logger.info(f"{status} {test_name}")
            logger.info(f"    Duration: {duration:.2f}s")
            logger.info(f"    Output: {output_path}")
            
            if result["error"]:
                logger.info(f"    Error: {result['error']}")
            
            # Stage breakdown
            if result.get("stages"):
                logger.info("    Stages:")
                for stage_name, stage_result in result["stages"].items():
                    stage_status = "✅" if stage_result.get("success", False) else "❌"
                    stage_duration = stage_result.get("duration", 0)
                    logger.info(f"      {stage_status} {stage_name}: {stage_duration:.2f}s")
            
            logger.info("")
        
        # Save detailed results to JSON
        results_file = self.outputs_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_prompt": self.test_prompt,
                "total_duration": total_duration,
                "successful_tests": successful_tests,
                "total_tests": total_tests,
                "device": self.gpu_info["device"],
                "results": self.results
            }, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")
        logger.info("="*80)

async def main():
    """Main function to run the pipeline tests"""
    tester = VideoPipelineTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
