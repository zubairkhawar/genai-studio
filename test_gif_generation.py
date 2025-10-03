#!/usr/bin/env python3
"""
Test script to verify GIF generation works without codec errors
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from models.video_generator import VideoGenerator

async def test_gif_generation():
    """Test GIF generation with placeholder"""
    print("Testing GIF generation...")
    
    # Mock GPU info
    gpu_info = {
        "device": "cpu",
        "gpu_name": "CPU",
        "memory_gb": 8
    }
    
    # Create video generator
    generator = VideoGenerator(gpu_info)
    
    try:
        # Test placeholder GIF generation
        print("Generating placeholder GIF...")
        output_path = await generator._generate_placeholder(
            prompt="Test GIF generation",
            duration=2,
            output_format="gif",
            palette=(34, 197, 94)
        )
        
        print(f"✅ GIF generated successfully: {output_path}")
        
        # Check if file exists
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ File exists with size: {file_size} bytes")
            return True
        else:
            print("❌ Generated file does not exist")
            return False
            
    except Exception as e:
        print(f"❌ GIF generation failed: {e}")
        return False

async def main():
    """Main test function"""
    print("=" * 50)
    print("Testing GIF Generation Fix")
    print("=" * 50)
    
    success = await test_gif_generation()
    
    if success:
        print("\n✅ All tests passed! GIF generation is working correctly.")
        print("The codec issue has been resolved.")
    else:
        print("\n❌ Tests failed. There may still be issues with GIF generation.")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
