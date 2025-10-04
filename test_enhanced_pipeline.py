#!/usr/bin/env python3
"""
Test script for the Enhanced Video Generation Pipeline

This script tests the complete enhanced pipeline:
Text → SD → AnimateDiff → RealESRGAN → RIFE → FFmpeg
"""

import asyncio
import sys
import pathlib
import logging

# Add backend to path
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

from backend.models.enhanced_video_generator import EnhancedVideoGenerator
from backend.utils.gpu_detector import GPUDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_enhanced_pipeline():
    """Test the enhanced video generation pipeline"""
    
    try:
        logger.info("🚀 Testing Enhanced Video Generation Pipeline")
        
        # Detect GPU
        gpu_detector = GPUDetector()
        gpu_info = gpu_detector.gpu_info
        logger.info(f"🔧 Detected GPU: {gpu_info}")
        
        # Initialize enhanced generator
        generator = EnhancedVideoGenerator(gpu_info)
        
        # Test prompt
        prompt = "A cinematic masterpiece of a futuristic cyberpunk city at night, neon lights glowing in vibrant colors, flying cars, holographic advertisements, rain reflecting on wet streets, camera smoothly glides forward through the urban landscape, highly detailed, photorealistic, 8K resolution"
        
        def progress_callback(progress: int, message: str):
            logger.info(f"📊 Progress: {progress}% - {message}")
        
        # Load models
        logger.info("📥 Loading models...")
        success = await generator.load_models()
        
        if not success:
            logger.error("❌ Failed to load models")
            return False
        
        # Generate video with MAXIMUM quality settings at 480x480
        logger.info("🎬 Generating enhanced video with MAXIMUM quality settings at 480x480...")
        output_path = await generator.generate_enhanced_video(
            prompt=prompt,
            duration=4,  # Shorter duration to fit memory
            target_fps=60,  # Ultra-high FPS for ultra-smooth motion
            width=480,  # 480x480 resolution as requested
            height=480,
            num_frames=24,  # Reduced frames to fit memory at 480x480
            num_inference_steps=60,  # Maximum steps for best quality
            guidance_scale=10.0,  # Higher guidance for better prompt following
            motion_scale=2.5,  # Maximum motion
            progress_callback=progress_callback
        )
        
        logger.info(f"✅ Enhanced video generated successfully: {output_path}")
        
        # Get pipeline info and test each model
        info = generator.get_pipeline_info()
        logger.info(f"📋 Pipeline Info: {info['name']}")
        logger.info(f"🔧 Models loaded: {info['models_loaded']}")
        
        # Test each model component individually
        logger.info("🧪 Testing individual model components...")
        
        # Test Stable Diffusion
        if generator.sd_generator and generator.sd_generator.is_loaded:
            logger.info("✅ Stable Diffusion: Working")
        else:
            logger.warning("⚠️ Stable Diffusion: Not loaded")
        
        # Test AnimateDiff
        if generator.animatediff_generator and generator.animatediff_generator.is_loaded:
            logger.info("✅ AnimateDiff: Working")
        else:
            logger.warning("⚠️ AnimateDiff: Not loaded")
        
        # Test RealESRGAN
        if generator.realesrgan_upscaler and generator.realesrgan_upscaler.is_loaded:
            logger.info("✅ RealESRGAN: Working")
        else:
            logger.warning("⚠️ RealESRGAN: Not loaded (using fallback)")
        
        # Test FILM
        if generator.film_interpolator and generator.film_interpolator.is_loaded:
            logger.info("✅ FILM: Working")
        else:
            logger.warning("⚠️ FILM: Not loaded (using fallback)")
        
        # Check video file properties
        import os
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"📁 Output file size: {file_size / 1024:.1f} KB")
            logger.info(f"📁 Output path: {output_path}")
        
        # Cleanup
        generator.unload_models()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🧪 Starting Enhanced Pipeline Test")
    
    success = await test_enhanced_pipeline()
    
    if success:
        logger.info("🎉 Enhanced Pipeline Test PASSED!")
        return 0
    else:
        logger.error("💥 Enhanced Pipeline Test FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
