#!/usr/bin/env python3
"""
Test script for AnimateDiff integration with existing Stable Diffusion model
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from models.animatediff_generator import AnimateDiffGenerator
from utils.gpu_detector import GPUDetector

async def test_animatediff_integration():
    """Test the AnimateDiff integration"""
    print("🚀 Testing AnimateDiff integration with existing Stable Diffusion model...")
    
    # Detect GPU
    gpu_detector = GPUDetector()
    gpu_info = gpu_detector.gpu_info
    print(f"📱 Device: {gpu_info['device']}")
    print(f"💾 Memory: {gpu_info.get('memory_gb', 'Unknown')} GB")
    
    # Initialize AnimateDiff generator
    generator = AnimateDiffGenerator(
        device=gpu_info["device"],
        dtype=gpu_info.get("dtype", "float16")
    )
    
    print("\n📋 Model Information:")
    info = generator.get_model_info()
    for key, value in info.items():
        if key != "config":
            print(f"  {key}: {value}")
    
    print("\n🔍 Verifying model files...")
    
    # Check if Stable Diffusion model exists
    project_root = Path(__file__).parent
    sd_path = project_root / "models" / "image" / "stable-diffusion"
    print(f"🔍 Checking SD path: {sd_path}")
    print(f"🔍 Path exists: {sd_path.exists()}")
    
    if sd_path.exists():
        sd_files = list(sd_path.rglob("*.safetensors")) + list(sd_path.rglob("*.bin"))
        print(f"✅ Stable Diffusion model found: {len(sd_files)} weight files")
    else:
        print("❌ Stable Diffusion model not found")
        return False
    
    # Check if motion adapter exists
    motion_path = project_root / "models" / "video" / "animatediff"
    print(f"🔍 Checking motion path: {motion_path}")
    print(f"🔍 Path exists: {motion_path.exists()}")
    
    if motion_path.exists():
        motion_files = list(motion_path.rglob("*.safetensors")) + list(motion_path.rglob("*.bin"))
        print(f"✅ Motion adapter found: {len(motion_files)} weight files")
    else:
        print("❌ Motion adapter not found")
        return False
    
    print("\n🔄 Loading AnimateDiff model...")
    try:
        success = await generator.load_model()
        if success:
            print("✅ AnimateDiff model loaded successfully!")
        else:
            print("❌ Failed to load AnimateDiff model")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    print("\n🎬 Testing video generation...")
    try:
        def progress_callback(progress, message):
            print(f"  Progress: {progress}% - {message}")
        
        # Generate a simple test video with MPS-optimized settings
        output_path = await generator.generate_video(
            prompt="beach with a sunset, cinematic, high quality, waves gently moving",
            width=256,  # Small resolution for MPS
            height=256,  # Small resolution for MPS
            num_frames=6,  # Short video for testing
            num_inference_steps=8,  # Few steps for faster testing
            progress_callback=progress_callback
        )
        
        print(f"✅ Video generated successfully: {output_path}")
        
        # Check if file exists
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            print(f"📁 File size: {file_size:.2f} MB")
            return True
        else:
            print("❌ Generated file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error generating video: {e}")
        return False
    
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        generator.unload_model()

async def main():
    """Main test function"""
    print("=" * 60)
    print("AnimateDiff Integration Test")
    print("=" * 60)
    
    success = await test_animatediff_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! AnimateDiff integration is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
