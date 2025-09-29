#!/usr/bin/env python3
"""
Test script for improved AnimateDiff quality settings
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from models.animatediff_generator import AnimateDiffGenerator
from utils.gpu_detector import GPUDetector

async def test_improved_animatediff():
    """Test the improved AnimateDiff settings"""
    print("🎬 Testing Improved AnimateDiff Quality Settings...")
    
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
    
    print("\n📋 Improved Configuration:")
    info = generator.get_model_info()
    config = info["config"]
    print(f"  Resolution: {config['width']}x{config['height']}")
    print(f"  Frames: {config['num_frames']}")
    print(f"  Inference Steps: {config['num_inference_steps']}")
    print(f"  Guidance Scale: {config['guidance_scale']}")
    print(f"  Motion Scale: {config['motion_scale']}")
    
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
    
    print("\n🎬 Testing improved video generation...")
    try:
        def progress_callback(progress, message):
            print(f"  Progress: {progress}% - {message}")
        
        # Test with the same prompt but improved settings
        output_path = await generator.generate_video(
            prompt="beach with a sunset, gentle ocean waves moving, slow camera pan, golden hour light, cinematic, high quality, detailed, masterpiece",
            negative_prompt="blurry, low quality, distorted, artifacts, bad anatomy, deformed, ugly, pixelated, low resolution, grainy, noise, compression artifacts, jpeg artifacts, watermark, text, signature",
            # Use default balanced settings for MPS
            progress_callback=progress_callback
        )
        
        print(f"✅ Improved video generated successfully: {output_path}")
        
        # Check if file exists
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            print(f"📁 File size: {file_size:.2f} MB")
            print(f"🎯 Expected improvements:")
            print(f"   - Higher resolution (384x384 vs 256x256)")
            print(f"   - More frames (12 vs 8)")
            print(f"   - More inference steps (20 vs 10)")
            print(f"   - Better scheduler (DDIM vs DPMSolver)")
            print(f"   - Negative prompt for quality control")
            print(f"   - Higher guidance scale (8.0 vs 7.5)")
            print(f"   - Increased motion scale (1.2 vs 1.0)")
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
    print("=" * 70)
    print("Improved AnimateDiff Quality Test")
    print("=" * 70)
    
    success = await test_improved_animatediff()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 Improved AnimateDiff test completed!")
        print("📈 The new settings should produce much clearer, higher quality videos.")
        print("🔧 Key improvements:")
        print("   • 1.5x higher resolution (384x384)")
        print("   • 1.5x more frames (12 frames)")
        print("   • 2x more inference steps (20 steps)")
        print("   • Better DDIM scheduler")
        print("   • Quality-focused negative prompts")
        print("   • Higher guidance scale (8.0)")
        print("   • Increased motion scale (1.2)")
    else:
        print("❌ Test failed. Check the output above for details.")
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
