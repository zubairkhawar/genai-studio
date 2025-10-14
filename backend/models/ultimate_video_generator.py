"""
Simple Video Generation Pipeline

Text → Stable Diffusion (keyframe) → Stable Video Diffusion (motion generation) → FFmpeg (assembly)

Supports 256x256, 384x384, and 512x512 resolutions. No upscaling - direct generation at selected resolution.
"""

import asyncio
import logging
import pathlib
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from .text_to_image import TextToImageGenerator
from .svd_generator import SVDGenerator

logger = logging.getLogger(__name__)


class UltimateVideoGenerator:
    def __init__(self, gpu_info: Dict[str, Any]):
        self.gpu_info = gpu_info
        self.device = gpu_info["device"]

        self.sd = None
        self.svd = None
        # Upscaling models removed

        project_root = pathlib.Path(__file__).parent.parent.parent
        self.outputs_dir = project_root / "outputs" / "videos"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.presets = self._build_presets()

    def _build_presets(self) -> Dict[str, Dict[str, Any]]:
        # Presets from user table
        return {
            "ultra-fast": {
                "sd": {"steps": 18, "cfg": 6.5, "sampler": "euler_a", "denoise": 0.35},
                "svd": {"frames": 16, "fps": 8, "motion_bucket_id": 127, "noise_aug_strength": 0.02, "steps": 12},
                "ffmpeg": {"filters": "eq=contrast=1.03:saturation=1.05,unsharp=2:2:0.2", "codec": "libx264", "crf": 21, "preset": "fast", "pix_fmt": "yuv420p"},
                "final": {"fps": 24},
            },
            "balanced": {
                "sd": {"steps": 30, "cfg": 7.0, "sampler": "dpmpp_2m", "denoise": 0.45},
                "svd": {"frames": 20, "fps": 8, "motion_bucket_id": 127, "noise_aug_strength": 0.02, "steps": 25},
                "ffmpeg": {"filters": "eq=contrast=1.05:saturation=1.1,unsharp=3:3:0.3", "codec": "libx264", "crf": 20, "preset": "medium", "pix_fmt": "yuv420p"},
                "final": {"fps": 24},
            },
            "high-quality": {
                "sd": {"steps": 35, "cfg": 7.5, "sampler": "dpmpp_sde", "denoise": 0.50},
                "svd": {"frames": 24, "fps": 8, "motion_bucket_id": 127, "noise_aug_strength": 0.02, "steps": 35},
                "ffmpeg": {"filters": "eq=contrast=1.07:saturation=1.15,unsharp=4:4:0.4", "codec": "libx264", "crf": 19, "preset": "slow", "pix_fmt": "yuv420p"},
                "final": {"fps": 24},
            },
        }

    async def load_models(self) -> bool:
        try:
            self.sd = TextToImageGenerator(device=self.device)
            project_root = pathlib.Path(__file__).parent.parent.parent
            sd_path = project_root / "models" / "image" / "stable-diffusion"
            if not await asyncio.to_thread(self.sd.load_model, str(sd_path)):
                return False

            self.svd = SVDGenerator(device=self.device)
            if not await self.svd.load_model():
                return False

            # Upscaling models removed - using direct SD → SVD pipeline

            return True
        except Exception as e:
            logger.error(f"Failed to load Ultimate pipeline models: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        preset: str,
        resolution: int = 512,  # Changed from target_resolution to resolution
        negative_prompt: Optional[str] = None,
        svd_overrides: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        output_format: str = "mp4",
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> str:
        if preset not in self.presets:
            raise ValueError("Unknown preset. Use 'ultra-fast', 'balanced', or 'high-quality'")

        cfg = self.presets[preset]

        # Use user-selected resolution for both SD and SVD
        base_resolution = resolution
        logger.info(f"Using resolution: {base_resolution}px for both SD and SVD")
        
        if progress_callback:
            progress_callback(5, f"Generating keyframe at {base_resolution}px (Stable Diffusion)")

        # Generate SD keyframe image at selected resolution
        keyframe: Image.Image = await asyncio.to_thread(
            self.sd.generate_image,
            prompt,
            base_resolution,
            base_resolution,
            cfg["sd"]["steps"],
            cfg["sd"]["cfg"],
            seed,
        )

        if progress_callback:
            progress_callback(20, f"Generating motion with SVD at {base_resolution}px")

        svd_cfg = cfg["svd"].copy()
        svd_overrides = svd_overrides or {}
        # Only allow SVD parameters to be overridden
        for k in ["frames", "fps", "motion_bucket_id", "noise_aug_strength", "steps"]:
            if k in svd_overrides:
                svd_cfg[k] = svd_overrides[k]

        # Use the same resolution for SVD as SD
        svd_width = base_resolution
        svd_height = base_resolution

        # Generate video at base resolution using SVD
        video_path = await self.svd.generate_video(
            init_image=keyframe,
            width=svd_width,
            height=svd_height,
            num_frames=svd_cfg["frames"],
            num_inference_steps=svd_cfg["steps"],
            motion_bucket_id=svd_cfg["motion_bucket_id"],
            noise_aug_strength=svd_cfg["noise_aug_strength"],
            seed=seed,
            output_format=output_format,
            progress_callback=lambda p, m: progress_callback(10 + int(p * 0.3), f"SVD: {m}") if progress_callback else None,
        )

        if progress_callback:
            progress_callback(40, "Extracting frames")

        frames = await self._extract_frames(video_path)
        # Remove intermediate SVD video to prevent duplicate UI entries
        try:
            import os
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Removed intermediate SVD video: {video_path}")
        except Exception as _:
            pass

        # Skip upscaling - use frames at original resolution
        if progress_callback:
            progress_callback(80, "Skipping upscaling - using original resolution")

        if progress_callback:
            progress_callback(92, "Final assembly")

        # For GIF format, use SVD's GIF method for better quality
        if output_format.lower() == "gif":
            output = await self._save_frames_as_gif(frames, prompt, cfg["final"]["fps"])
        else:
            output = await self._ffmpeg_assemble(
                frames,
                prompt,
                cfg["ffmpeg"],
                cfg["final"]["fps"],
                (base_resolution, base_resolution),  # Use original resolution for final output
                output_format
            )
        if progress_callback:
            progress_callback(100, "Done")
        return output


    async def _clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM errors"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info("GPU memory cleared")
        except Exception as e:
            logger.warning(f"Could not clear GPU memory: {e}")

    async def _extract_frames(self, video_path: str) -> List[Image.Image]:
        import cv2
        frames: List[Image.Image] = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames

    async def _ffmpeg_assemble(
        self,
        frames: List[Image.Image],
        prompt: str,
        ffmpeg_cfg: Dict[str, Any],
        fps: int,
        final_size: tuple,
        output_format: str = "mp4",
    ) -> str:
        import tempfile
        import shutil
        import subprocess

        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            for i, f in enumerate(frames):
                p = tmp / f"f_{i:06d}.png"
                # Downsample to final_size for user output
                if final_size:
                    f = f.resize(final_size, Image.BICUBIC)
                f.save(p, "PNG")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.outputs_dir / f"video_{ts}.{output_format}"

            cmd = [
                "ffmpeg","-y",
                "-framerate", str(fps),
                "-i", str(tmp / "f_%06d.png"),
                "-vf", ffmpeg_cfg["filters"],
            ]
            
            # Add format-specific options
            if output_format.lower() == "mp4":
                cmd.extend([
                    "-c:v", ffmpeg_cfg["codec"],
                    "-pix_fmt", ffmpeg_cfg["pix_fmt"],
                    "-crf", str(ffmpeg_cfg["crf"]),
                    "-preset", ffmpeg_cfg["preset"],
                    "-movflags", "+faststart",
                ])
            elif output_format.lower() == "gif":
                # For GIF, use different settings
                cmd.extend([
                    "-vf", f"{ffmpeg_cfg['filters']},palettegen=reserve_transparent=0",
                    "-c:v", "gif",
                ])
            
            cmd.extend([
                "-metadata", f"title=Video: {prompt[:50]}",
                str(out_path),
            ])
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, err = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(err.decode())
            return str(out_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def _save_frames_as_gif(self, frames: List[Image.Image], prompt: str, fps: int) -> str:
        """Save frames as GIF using PIL for better quality"""
        try:
            # Calculate duration based on FPS
            duration = int(1000 / fps)  # Convert FPS to milliseconds per frame
            
            # Generate filename
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{ts}.gif"
            output_path = self.outputs_dir / filename
            
            # Save the first frame
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,  # Infinite loop
                optimize=True,
                comment=f"Video: {prompt[:50]}"
            )
            
            logger.info(f"✅ GIF saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save GIF: {e}")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available video models"""
        return [
            {
                "id": "svd",
                "name": "Stable Video Diffusion",
                "description": "Image-to-video generation using Stable Video Diffusion",
                "max_duration": 5,
                "resolution": "256x256, 384x384, 512x512",
                "loaded": self.sd is not None and self.svd is not None,
                "size_gb": 32.61
            }
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "Stable Video Diffusion",
            "description": "Image-to-video generation using Stable Video Diffusion",
            "loaded": self.sd is not None and self.svd is not None,
            "device": self.device,
            "available_models": self.get_available_models(),
            "capabilities": [
                "text-to-video",
                "image-to-video", 
                "MP4 generation",
                "multiple resolutions (256x256, 384x384, 512x512)",
                "local model integration"
            ]
        }

    async def load_default_models(self) -> bool:
        """Load default models for the pipeline"""
        try:
            return await self.load_models()
        except Exception as e:
            logger.error(f"Failed to load default models: {e}")
            return False


