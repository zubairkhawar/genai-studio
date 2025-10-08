#!/usr/bin/env python3
import asyncio
import pathlib
import sys
import logging
import os

# Set MPS memory limit to no cap for testing
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Ensure backend is importable
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

from backend.utils.gpu_detector import GPUDetector
from backend.models.animatediff_generator import AnimateDiffGenerator
from backend.models.text_to_image import TextToImageGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPT = "Ocean waves crashing against rocky cliffs, cinematic lighting"
NEG = "blurry, low quality, artifacts, static"

async def main():
    gpu = GPUDetector().gpu_info
    print(f"Device: {gpu['device']}")
    
    # Step 1: Generate SD keyframe
    print("Step 1: Generating SD keyframe...")
    sd_gen = TextToImageGenerator(device=gpu['device'])
    sd_path = pathlib.Path(__file__).parent / "models" / "image" / "stable-diffusion"
    sd_loaded = await asyncio.to_thread(sd_gen.load_model, str(sd_path))
    if not sd_loaded:
        print("Failed to load SD")
        return
    
    keyframe = await asyncio.to_thread(
        sd_gen.generate_image,
        prompt=PROMPT,
        width=384,  # Lower resolution
        height=384,
        num_inference_steps=15,  # Fewer steps
        guidance_scale=7.0
    )
    print(f"✅ SD keyframe generated: {keyframe.width}x{keyframe.height}")
    
    # Step 2: Generate AnimateDiff video with keyframe
    print("Step 2: Generating AnimateDiff video with keyframe...")
    ad_gen = AnimateDiffGenerator(device=gpu['device'])
    ad_loaded = await ad_gen.load_model()
    if not ad_loaded:
        print("Failed to load AnimateDiff")
        return
    
    def progress_callback(progress, message):
        print(f"  {progress}%: {message}")
    
    video_path = await ad_gen.generate_video(
        prompt=PROMPT,
        negative_prompt=NEG,
        init_image=keyframe,  # Pass the SD keyframe
        width=256,  # Very low resolution for memory
        height=256,
        num_frames=12,  # More frames for smoother motion
        num_inference_steps=15,  # More steps for better quality
        guidance_scale=8.0,  # Higher guidance for better prompt adherence
        motion_scale=2.5,  # Higher motion for more visible movement
        seed=42,
        output_format="mp4",
        progress_callback=progress_callback
    )
    
    print(f"✅ Complete pipeline output: {video_path}")
    print("Pipeline: Text → SD → AnimateDiff → FFmpeg ✅")

if __name__ == "__main__":
    asyncio.run(main())
