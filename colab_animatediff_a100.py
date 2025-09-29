# Google Colab Notebook for High-Quality AnimateDiff Video Generation on A100
# Run this in Google Colab with A100 GPU runtime

# Install required packages
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q diffusers transformers accelerate xformers
!pip install -q opencv-python pillow imageio imageio-ffmpeg
!pip install -q huggingface_hub

# Import libraries
import torch
import os
import numpy as np
from PIL import Image
import cv2
from diffusers import StableDiffusionPipeline, MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
import imageio
import tempfile
import shutil
from pathlib import Path

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "No GPU")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Using device: {device} with dtype: {dtype}")

# High-quality configuration for A100
config = {
    "width": 512,           # High resolution
    "height": 512,          # High resolution  
    "num_frames": 24,       # 3 seconds at 8 FPS
    "num_inference_steps": 30,  # High quality
    "guidance_scale": 7.5,  # Good prompt adherence
    "motion_scale": 1.5,    # Strong motion
    "fps": 8,
    "seed": 42
}

print("High-Quality Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Download models
print("\n📥 Downloading models...")

# Download Stable Diffusion v1.5
print("Downloading Stable Diffusion v1.5...")
sd_model_id = "runwayml/stable-diffusion-v1-5"
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    sd_model_id,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False
)

# Download AnimateDiff motion adapter
print("Downloading AnimateDiff motion adapter...")
motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-2"
motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=dtype)

# Create AnimateDiff pipeline
print("Creating AnimateDiff pipeline...")
pipeline = AnimateDiffPipeline.from_pretrained(
    sd_model_id,
    motion_adapter=motion_adapter,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False
)

# Move to GPU and optimize
pipeline = pipeline.to(device)

# Enable optimizations for A100
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()

# Use DDIM scheduler for better temporal consistency
pipeline.scheduler = DDIMScheduler.from_config(
    pipeline.scheduler.config,
    clip_sample=False,
    timestep_spacing="linspace",
    steps_offset=1,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    prediction_type="epsilon"
)

print("✅ Models loaded and optimized!")

# Test prompt
test_prompt = "A majestic eagle soaring through mountain peaks at sunset, cinematic lighting, 4K quality, detailed feathers, dramatic sky"
negative_prompt = "blurry, low quality, distorted, artifacts, bad anatomy, deformed, ugly, pixelated, low resolution, grainy, noise, compression artifacts, jpeg artifacts, watermark, text, signature, inconsistent motion, flickering, frame drops, temporal artifacts, motion blur, choppy animation, unstable frames, inconsistent lighting, color shifts, frame interpolation errors"

print(f"\n🎬 Generating video with prompt: '{test_prompt}'")
print("This may take 2-5 minutes on A100...")

# Generate video
generator = torch.Generator(device=device).manual_seed(config["seed"])

with torch.autocast(device):
    result = pipeline(
        prompt=test_prompt,
        negative_prompt=negative_prompt,
        width=config["width"],
        height=config["height"],
        num_frames=config["num_frames"],
        num_inference_steps=config["num_inference_steps"],
        guidance_scale=config["guidance_scale"],
        motion_scale=config["motion_scale"],
        generator=generator
    )

print("✅ Video generation completed!")

# Get frames
frames = result.frames[0]
print(f"Generated {len(frames)} frames")

# Create output directory
output_dir = Path("/content/outputs")
output_dir.mkdir(exist_ok=True)

# Save frames as individual images first
frame_dir = output_dir / "frames"
frame_dir.mkdir(exist_ok=True)

print("💾 Saving frames...")
for i, frame in enumerate(frames):
    frame_path = frame_dir / f"frame_{i:04d}.png"
    frame.save(frame_path)

# Create video using imageio
print("🎥 Creating MP4 video...")
video_path = output_dir / "animatediff_high_quality.mp4"

# Convert PIL images to numpy arrays
frame_arrays = []
for frame in frames:
    frame_array = np.array(frame)
    frame_arrays.append(frame_array)

# Create video with imageio
imageio.mimsave(
    video_path,
    frame_arrays,
    fps=config["fps"],
    quality=8,
    macro_block_size=1
)

# Get video info
file_size = video_path.stat().st_size / (1024 * 1024)  # MB
print(f"📁 Video saved: {video_path}")
print(f"📊 File size: {file_size:.1f} MB")
print(f"📐 Resolution: {config['width']}x{config['height']}")
print(f"🎞️ Frames: {config['num_frames']} ({config['num_frames']/config['fps']:.1f}s)")
print(f"🎯 Quality: High (30 inference steps)")

# Display first frame
print("\n🖼️ First frame preview:")
display(frames[0])

# Create download link
from google.colab import files
print("\n📥 Download your video:")
files.download(str(video_path))

# Also create a GIF version for preview
print("🎞️ Creating GIF preview...")
gif_path = output_dir / "animatediff_preview.gif"
imageio.mimsave(
    gif_path,
    frame_arrays,
    fps=4,  # Slower for GIF
    duration=0.25
)

print(f"📁 GIF preview saved: {gif_path}")
files.download(str(gif_path))

print("\n🎉 High-quality video generation completed!")
print("You can download both the MP4 video and GIF preview above.")
print(f"Video quality: {config['width']}x{config['height']} @ {config['fps']} FPS for {config['num_frames']/config['fps']:.1f} seconds")
