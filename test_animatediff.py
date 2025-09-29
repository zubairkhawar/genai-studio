#!/usr/bin/env python3
"""
Test script for the new AnimateDiff setup
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from models.animatediff_generator import AnimateDiffGenerator

async def test_animatediff():
    """Test the new AnimateDiff generator"""
    print("🎬 Testing AnimateDiff Generator...")
    
    # Initialize generator
    generator = AnimateDiffGenerator(device="cpu")  # Use CPU for testing
    
    # Test model loading
    print("📥 Loading AnimateDiff model...")
    success = await generator.load_model()
    
    if success:
        print("✅ AnimateDiff model loaded successfully!")
        
        # Get model info
        info = generator.get_model_info()
        print(f"📊 Model Info:")
        print(f"   Name: {info['name']}")
        print(f"   Available models: {info['available_models']}")
        print(f"   AnimateDiff dir: {info['animatediff_dir']}")
        print(f"   Motion adapter dir: {info['motion_adapter_dir']}")
        
        # Test available models
        models = generator.get_available_models()
        print(f"🎨 Available community models: {models}")
        
    else:
        print("❌ Failed to load AnimateDiff model")
        print("💡 Make sure to run the download script first:")
        print("   python scripts/download-models.py --model animatediff")
    
    return success

if __name__ == "__main__":
    asyncio.run(test_animatediff())


