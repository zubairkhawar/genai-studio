#!/usr/bin/env python3
"""
Test video generation with hardware-optimized settings
"""

import asyncio
import sys
import os
import pathlib
import logging

# Add backend to path
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

from backend.utils.gpu_detector import GPUDetector
from backend.models.video_generator import VideoGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_video_generation():
    """Test video generation with a sample prompt"""
    print("🎬 Testing Video Generation with Hardware-Optimized Settings")
    print("=" * 60)
    
    # Detect GPU
    gpu_detector = GPUDetector()
    gpu_info = gpu_detector.detect_gpu()
    
    print(f"Detected Hardware: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
    print(f"Device: {gpu_info['device']}")
    
    # Initialize video generator
    video_generator = VideoGenerator(gpu_info)
    
    # Test prompt
    test_prompt = "A beautiful sunset over a calm ocean with gentle waves, cinematic lighting, 4K quality"
    
    print(f"\n🎯 Test Prompt: '{test_prompt}'")
    print("Loading AnimateDiff model...")
    
    try:
        # Load the AnimateDiff model
        success = await video_generator.load_model("animatediff")
        if not success:
            print("❌ Failed to load AnimateDiff model")
            return None
        
        print("✅ Model loaded successfully")
        
        # Generate video
        print("Generating video with hardware-optimized settings...")
        
        def progress_callback(progress, message):
            print(f"Progress: {progress}% - {message}")
        
        output_path = await video_generator.generate(
            prompt=test_prompt,
            model_name="animatediff",
            duration=4,  # 4 seconds requested
            output_format="mp4",
            progress_callback=progress_callback
        )
        
        print(f"\n✅ Video generated successfully!")
        print(f"📁 Output: {output_path}")
        
        # Get file size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"📊 File Size: {file_size:.1f} MB")
            
            # Get file info
            import subprocess
            try:
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', output_path
                ], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    import json
                    info = json.loads(result.stdout)
                    if 'streams' in info and len(info['streams']) > 0:
                        stream = info['streams'][0]
                        width = stream.get('width', 'Unknown')
                        height = stream.get('height', 'Unknown')
                        duration = stream.get('duration', 'Unknown')
                        fps = stream.get('r_frame_rate', 'Unknown')
                        print(f"📐 Resolution: {width}x{height}")
                        print(f"⏱️  Duration: {duration}s")
                        print(f"🎞️  Frame Rate: {fps}")
            except Exception as e:
                print(f"Could not get detailed video info: {e}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    try:
        output_path = await test_video_generation()
        
        if output_path:
            print("\n🎉 Video generation test completed successfully!")
            print(f"Generated video: {output_path}")
            print("\n💡 You can now view the video in your outputs directory")
        else:
            print("\n❌ Video generation test failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
