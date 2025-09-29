#!/usr/bin/env python3
"""
Test script for hardware-optimized video generation
Demonstrates automatic hardware detection and optimized settings
"""

import asyncio
import sys
import os
import pathlib
import logging

# Add backend to path
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

from backend.utils.gpu_detector import GPUDetector
from backend.models.animatediff_generator import AnimateDiffGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_hardware_detection():
    """Test hardware detection and configuration"""
    print("🔍 Testing Hardware Detection...")
    print("=" * 50)
    
    # Detect GPU
    gpu_detector = GPUDetector()
    gpu_info = gpu_detector.detect_gpu()
    
    print(f"GPU Type: {gpu_info['type']}")
    print(f"Device: {gpu_info['device']}")
    print(f"GPU Name: {gpu_info['name']}")
    print(f"Memory: {gpu_info['memory_gb']:.1f} GB")
    print(f"CUDA Available: {gpu_info.get('cuda_available', False)}")
    print(f"ROCm Available: {gpu_info.get('rocm_available', False)}")
    
    return gpu_info

async def test_animatediff_configuration(gpu_info):
    """Test AnimateDiff configuration with hardware detection"""
    print("\n🎬 Testing AnimateDiff Configuration...")
    print("=" * 50)
    
    # Initialize AnimateDiff with hardware detection
    device = gpu_info["device"]
    dtype = "float32" if device == "mps" else "float16"
    
    generator = AnimateDiffGenerator(device=device, dtype=dtype)
    
    # Show detected configuration
    config = generator.config
    print(f"Hardware-Optimized Configuration:")
    print(f"  Resolution: {config['width']}x{config['height']}")
    print(f"  Frames: {config['num_frames']} ({config['num_frames']/8:.1f}s at 8 FPS)")
    print(f"  Inference Steps: {config['num_inference_steps']}")
    print(f"  Guidance Scale: {config['guidance_scale']}")
    print(f"  Motion Scale: {config['motion_scale']}")
    print(f"  Precision: {config.get('precision', 'auto')}")
    print(f"  Attention Slicing: {config.get('attention_slicing', True)}")
    print(f"  VAE Tiling: {config.get('vae_tiling', False)}")
    print(f"  Memory Fraction: {config.get('memory_fraction', 1.0)}")
    
    return generator

async def test_video_generation(generator):
    """Test video generation with optimized settings"""
    print("\n🎥 Testing Video Generation...")
    print("=" * 50)
    
    # Test prompt
    test_prompt = "A majestic eagle soaring through mountain peaks at sunset, cinematic lighting, 4K quality"
    
    print(f"Test Prompt: '{test_prompt}'")
    print("Loading AnimateDiff model...")
    
    try:
        # Load model
        success = await generator.load_model()
        if not success:
            print("❌ Failed to load AnimateDiff model")
            return None
        
        print("✅ Model loaded successfully")
        
        # Generate video
        print("Generating video with hardware-optimized settings...")
        
        def progress_callback(progress, message):
            print(f"Progress: {progress}% - {message}")
        
        output_path = await generator.generate_video(
            prompt=test_prompt,
            progress_callback=progress_callback
        )
        
        print(f"✅ Video generated successfully!")
        print(f"Output: {output_path}")
        
        # Get file size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"File Size: {file_size:.1f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        return None

async def main():
    """Main test function"""
    print("🚀 Hardware-Optimized Video Generation Test")
    print("=" * 60)
    
    try:
        # Test 1: Hardware Detection
        gpu_info = await test_hardware_detection()
        
        # Test 2: Configuration
        generator = await test_animatediff_configuration(gpu_info)
        
        # Test 3: Video Generation
        output_path = await test_video_generation(generator)
        
        if output_path:
            print("\n🎉 Test completed successfully!")
            print(f"Generated video: {output_path}")
        else:
            print("\n❌ Test failed - no video generated")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
