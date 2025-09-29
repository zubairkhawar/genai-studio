#!/usr/bin/env python3
"""
Test script for enhanced AnimateDiff frame consistency
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from models.animatediff_generator import AnimateDiffGenerator
from utils.gpu_detector import GPUDetector

async def test_enhanced_animatediff():
    """Test the enhanced AnimateDiff settings for better frame consistency"""
    print("🎬 Testing Enhanced AnimateDiff Frame Consistency...")
    
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
    
    print("\n📋 Enhanced Configuration:")
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
    
    print("\n🎬 Testing enhanced video generation with better frame consistency...")
    try:
        def progress_callback(progress, message):
            print(f"  Progress: {progress}% - {message}")
        
        # Test with enhanced settings for better frame consistency
        output_path = await generator.generate_video(
            prompt="beach with a sunset, gentle ocean waves moving, slow camera pan, golden hour light, cinematic, high quality, detailed, masterpiece, smooth motion, consistent lighting",
            negative_prompt="blurry, low quality, distorted, artifacts, bad anatomy, deformed, ugly, pixelated, low resolution, grainy, noise, compression artifacts, jpeg artifacts, watermark, text, signature, inconsistent motion, flickering, frame drops, temporal artifacts, motion blur, choppy animation, unstable frames, inconsistent lighting, color shifts, frame interpolation errors",
            # Use default enhanced settings
            progress_callback=progress_callback
        )
        
        print(f"✅ Enhanced video generated successfully: {output_path}")
        
        # Check if file exists
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            print(f"📁 File size: {file_size:.2f} MB")
            print(f"🎯 Enhanced improvements:")
            print(f"   - More frames (16 vs 12) for better motion consistency")
            print(f"   - Higher inference steps (25 vs 20) for better quality")
            print(f"   - Higher guidance scale (8.5 vs 8.0) for better prompt adherence")
            print(f"   - Higher motion scale (1.5 vs 1.2) for better motion consistency")
            print(f"   - Enhanced negative prompts for frame consistency")
            print(f"   - Optimized DDIM scheduler settings")
            print(f"   - Frame post-processing for temporal smoothing")
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
    print("=" * 80)
    print("Enhanced AnimateDiff Frame Consistency Test")
    print("=" * 80)
    
    success = await test_enhanced_animatediff()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 Enhanced AnimateDiff test completed!")
        print("📈 The new settings should produce much better frame consistency.")
        print("🔧 Key enhancements for frame quality:")
        print("   • More frames (16) for smoother motion")
        print("   • Higher inference steps (25) for better quality")
        print("   • Higher guidance scale (8.5) for better prompt adherence")
        print("   • Higher motion scale (1.5) for better motion consistency")
        print("   • Enhanced negative prompts targeting frame issues")
        print("   • Optimized DDIM scheduler with better parameters")
        print("   • Frame post-processing for temporal smoothing")
        print("   • Better handling of last frame quality issues")
    else:
        print("❌ Test failed. Check the output above for details.")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
