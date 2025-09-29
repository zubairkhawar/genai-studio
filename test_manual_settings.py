#!/usr/bin/env python3
"""
Test script for manual video generation settings
"""

import asyncio
import sys
import pathlib
import logging

# Add backend to path
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

from backend.utils.gpu_detector import GPUDetector
from backend.models.video_generator import VideoGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_manual_settings():
    """Test video generation with manual settings"""
    print("🎬 Testing Manual Video Generation Settings")
    print("=" * 50)
    
    # Detect GPU
    gpu_detector = GPUDetector()
    gpu_info = gpu_detector.detect_gpu()
    
    print(f"Detected Hardware: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
    
    # Initialize video generator
    video_generator = VideoGenerator(gpu_info)
    
    # Test prompt
    test_prompt = "A beautiful sunset over a calm ocean with gentle waves, cinematic lighting"
    
    print(f"\n🎯 Test Prompt: '{test_prompt}'")
    
    # Test different manual settings
    test_configs = [
        {
            "name": "Ultra Fast",
            "settings": {
                "width": 256,
                "height": 256,
                "num_frames": 8,
                "num_inference_steps": 15,
                "guidance_scale": 7.0,
                "motion_scale": 1.2,
                "fps": 6,
                "seed": 42
            }
        },
        {
            "name": "Balanced",
            "settings": {
                "width": 384,
                "height": 384,
                "num_frames": 16,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "motion_scale": 1.4,
                "fps": 8,
                "seed": 123
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n🧪 Testing {config['name']} Configuration:")
        print(f"  Resolution: {config['settings']['width']}x{config['settings']['height']}")
        print(f"  Frames: {config['settings']['num_frames']}")
        print(f"  Inference Steps: {config['settings']['num_inference_steps']}")
        print(f"  Guidance Scale: {config['settings']['guidance_scale']}")
        print(f"  Motion Scale: {config['settings']['motion_scale']}")
        print(f"  FPS: {config['settings']['fps']}")
        print(f"  Seed: {config['settings']['seed']}")
        
        try:
            # Load model
            success = await video_generator.load_model("animatediff")
            if not success:
                print("❌ Failed to load AnimateDiff model")
                continue
            
            print("✅ Model loaded successfully")
            
            # Generate video with manual settings
            def progress_callback(progress, message):
                print(f"Progress: {progress}% - {message}")
            
            output_path = await video_generator.generate(
                prompt=test_prompt,
                model_name="animatediff",
                duration=2,  # Duration in seconds
                output_format="mp4",
                progress_callback=progress_callback,
                **config['settings']  # Pass manual settings
            )
            
            print(f"✅ {config['name']} video generated successfully!")
            print(f"📁 Output: {output_path}")
            
        except Exception as e:
            print(f"❌ {config['name']} generation failed: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main test function"""
    try:
        await test_manual_settings()
        print("\n🎉 Manual settings test completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
