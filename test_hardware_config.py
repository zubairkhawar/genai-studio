#!/usr/bin/env python3
"""
Test script to show hardware-optimized configuration without generating video
"""

import sys
import pathlib
import logging

# Add backend to path
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

from backend.utils.gpu_detector import GPUDetector
from backend.models.animatediff_generator import AnimateDiffGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hardware_detection():
    """Test hardware detection and configuration"""
    print("🔍 Hardware Detection Results")
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

def test_animatediff_configuration(gpu_info):
    """Test AnimateDiff configuration with hardware detection"""
    print("\n🎬 Hardware-Optimized Configuration")
    print("=" * 50)
    
    # Initialize AnimateDiff with hardware detection
    device = gpu_info["device"]
    dtype = "float32" if device == "mps" else "float16"
    
    generator = AnimateDiffGenerator(device=device, dtype=dtype)
    
    # Show detected configuration
    config = generator.config
    print(f"Detected Hardware: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
    print(f"Optimized Settings:")
    print(f"  📐 Resolution: {config['width']}x{config['height']}")
    print(f"  🎞️  Frames: {config['num_frames']} ({config['num_frames']/8:.1f}s at 8 FPS)")
    print(f"  🔄 Inference Steps: {config['num_inference_steps']}")
    print(f"  🎯 Guidance Scale: {config['guidance_scale']}")
    print(f"  🌊 Motion Scale: {config['motion_scale']}")
    print(f"  💾 Precision: {config.get('precision', 'auto')}")
    print(f"  ⚡ Attention Slicing: {config.get('attention_slicing', True)}")
    print(f"  🧩 VAE Tiling: {config.get('vae_tiling', False)}")
    print(f"  📊 Memory Fraction: {config.get('memory_fraction', 1.0)}")
    
    # Show expected performance
    print(f"\n📈 Expected Performance:")
    if device == "mps":
        if gpu_info['memory_gb'] >= 16:
            print(f"  🚀 High-quality mode for Apple Silicon")
            print(f"  ⏱️  Generation time: ~60-120 seconds")
            print(f"  📁 Output size: ~2-5 MB")
        else:
            print(f"  ⚖️  Balanced mode for lower memory")
            print(f"  ⏱️  Generation time: ~45-90 seconds")
            print(f"  📁 Output size: ~1-3 MB")
    elif device == "cuda":
        if gpu_info['memory_gb'] >= 20:
            print(f"  🔥 Maximum quality mode for high-VRAM GPU")
            print(f"  ⏱️  Generation time: ~30-60 seconds")
            print(f"  📁 Output size: ~5-15 MB")
        else:
            print(f"  ⚡ High-quality mode for standard GPU")
            print(f"  ⏱️  Generation time: ~45-90 seconds")
            print(f"  📁 Output size: ~2-5 MB")
    else:
        print(f"  🐌 CPU mode - slower but functional")
        print(f"  ⏱️  Generation time: ~5-15 minutes")
        print(f"  📁 Output size: ~1-2 MB")
    
    return generator

def show_comparison():
    """Show comparison between different hardware configurations"""
    print("\n📊 Hardware Configuration Comparison")
    print("=" * 60)
    
    configs = {
        "MacBook M1 (16GB)": {
            "resolution": "512x512",
            "frames": "16",
            "steps": "25",
            "guidance": "7.8",
            "motion": "1.5",
            "precision": "float32",
            "duration": "2.0s"
        },
        "AMD 7900 XTX (24GB)": {
            "resolution": "768x768", 
            "frames": "36",
            "steps": "38",
            "guidance": "7.2",
            "motion": "1.6",
            "precision": "float16",
            "duration": "4.5s"
        },
        "Standard GPU (8GB)": {
            "resolution": "512x512",
            "frames": "16", 
            "steps": "25",
            "guidance": "7.5",
            "motion": "1.4",
            "precision": "float16",
            "duration": "2.0s"
        }
    }
    
    print(f"{'Hardware':<20} {'Resolution':<12} {'Frames':<8} {'Steps':<6} {'Duration':<8} {'Quality'}")
    print("-" * 70)
    
    for hardware, config in configs.items():
        quality = "🔥 High" if "7900" in hardware else "⚡ Good" if "M1" in hardware else "📊 Standard"
        print(f"{hardware:<20} {config['resolution']:<12} {config['frames']:<8} {config['steps']:<6} {config['duration']:<8} {quality}")

def main():
    """Main test function"""
    print("🚀 Hardware-Optimized Video Generation Configuration")
    print("=" * 60)
    
    try:
        # Test 1: Hardware Detection
        gpu_info = test_hardware_detection()
        
        # Test 2: Configuration
        generator = test_animatediff_configuration(gpu_info)
        
        # Test 3: Comparison
        show_comparison()
        
        print("\n🎉 Configuration test completed successfully!")
        print("\n💡 To generate a video with these optimized settings:")
        print("   1. Use the frontend at http://localhost:3000/video")
        print("   2. Or call the API endpoint /generate with model_type='video'")
        print("   3. The system will automatically use these hardware-optimized settings")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
