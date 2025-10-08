#!/usr/bin/env python3
import asyncio
import pathlib
import sys
import logging

# Ensure backend is importable
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

from backend.utils.gpu_detector import GPUDetector
from backend.models.animatediff_generator import AnimateDiffGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPT = "Ocean waves crashing against rocky cliffs, cinematic lighting"
NEG = "blurry, low quality, artifacts, text"

async def main():
    gpu = GPUDetector().gpu_info
    print(f"Device: {gpu['device']}")
    gen = AnimateDiffGenerator(device=gpu['device'])
    ok = await gen.load_model()
    if not ok:
        print("Failed to load AnimateDiff")
        return
    try:
        # Run generation
        out = await gen.generate_video(
            prompt=PROMPT,
            negative_prompt=NEG,
            num_frames=16,
            num_inference_steps=25,
            guidance_scale=8.0,
            motion_scale=2.0,
            seed=42,
            output_format="mp4",
            progress_callback=lambda p, m: print(f"{p}% {m}")
        )
        print(f"Output: {out}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
