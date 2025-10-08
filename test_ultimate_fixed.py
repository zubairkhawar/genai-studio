#!/usr/bin/env python3
import asyncio
import pathlib
import sys
import logging

# Ensure backend is importable
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

from backend.utils.gpu_detector import GPUDetector
from backend.models.ultimate_video_generator import UltimateVideoGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPT = "Ocean waves crashing against rocky cliffs, cinematic lighting, dramatic motion"
NEG = "blurry, low quality, artifacts, static, still"

async def main():
    gpu = GPUDetector().gpu_info
    print(f"Device: {gpu['device']}")
    
    gen = UltimateVideoGenerator(gpu_info=gpu)
    ok = await gen.load_models()
    if not ok:
        print("Failed to load Ultimate Video Generator models")
        return
    
    try:
        def progress_callback(progress, message):
            print(f"{progress}%: {message}")
        
        out = await gen.generate(
            prompt=PROMPT,
            preset="balanced",
            negative_prompt=NEG,
            ad_overrides={
                "frames": 16,
                "motion_scale": 2.0,  # Higher motion for more movement
                "steps": 25
            },
            seed=42,
            progress_callback=progress_callback
        )
        print(f"✅ Output: {out}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
