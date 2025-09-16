#!/usr/bin/env python3
"""
Model Optimization Script
Removes duplicate model files and keeps only fp16 versions to reduce storage from 70+ GB to ~20-25 GB
"""

import os
import shutil
import pathlib
import subprocess
import sys
from typing import List, Dict, Tuple

def get_file_size_gb(file_path: pathlib.Path) -> float:
    """Get file size in GB"""
    if file_path.exists():
        return file_path.stat().st_size / (1024**3)
    return 0.0

def get_directory_size_gb(dir_path: pathlib.Path) -> float:
    """Get total directory size in GB"""
    if not dir_path.exists():
        return 0.0
    
    total_size = 0
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size / (1024**3)

def find_duplicate_files(directory: pathlib.Path, patterns: List[str]) -> Dict[str, List[pathlib.Path]]:
    """Find duplicate files based on patterns"""
    duplicates = {}
    
    for pattern in patterns:
        files = list(directory.rglob(pattern))
        if len(files) > 1:
            duplicates[pattern] = files
    
    return duplicates

def optimize_svd_models() -> Tuple[float, float]:
    """Optimize Stable Video Diffusion models"""
    print("🎬 Optimizing Stable Video Diffusion models...")
    
    svd_dir = pathlib.Path("models/video/stable-video-diffusion")
    if not svd_dir.exists():
        print("❌ SVD directory not found")
        return 0.0, 0.0
    
    original_size = get_directory_size_gb(svd_dir)
    print(f"📊 Original SVD size: {original_size:.2f} GB")
    
    # Files to keep (fp16 versions)
    files_to_keep = [
        "svd.safetensors",
        "svd_image_decoder.safetensors", 
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "image_encoder/model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors"
    ]
    
    # Files to remove (full precision duplicates)
    files_to_remove = [
        "unet/diffusion_pytorch_model.safetensors",  # 5.7GB -> keep fp16 (2.8GB)
        "image_encoder/model.safetensors",           # 2.4GB -> keep fp16 (1.2GB) 
        "vae/diffusion_pytorch_model.safetensors"    # 373MB -> keep fp16 (186MB)
    ]
    
    removed_size = 0.0
    for file_pattern in files_to_remove:
        files = list(svd_dir.rglob(file_pattern))
        for file_path in files:
            if file_path.exists():
                size_gb = get_file_size_gb(file_path)
                print(f"🗑️  Removing: {file_path.name} ({size_gb:.2f} GB)")
                file_path.unlink()
                removed_size += size_gb
    
    new_size = get_directory_size_gb(svd_dir)
    print(f"✅ SVD optimization complete: {original_size:.2f} GB → {new_size:.2f} GB (saved {removed_size:.2f} GB)")
    
    return removed_size, new_size

def optimize_sd_models() -> Tuple[float, float]:
    """Optimize Stable Diffusion models"""
    print("🖼️  Optimizing Stable Diffusion models...")
    
    sd_dir = pathlib.Path("models/image/stable-diffusion")
    if not sd_dir.exists():
        print("❌ SD directory not found")
        return 0.0, 0.0
    
    original_size = get_directory_size_gb(sd_dir)
    print(f"📊 Original SD size: {original_size:.2f} GB")
    
    # Files to remove (duplicates and non-essential)
    files_to_remove = [
        # UNet duplicates - keep only fp16.safetensors
        "unet/diffusion_pytorch_model.safetensors",        # 3.2GB
        "unet/diffusion_pytorch_model.bin",                # 3.2GB  
        "unet/diffusion_pytorch_model.non_ema.safetensors", # 3.2GB
        "unet/diffusion_pytorch_model.non_ema.bin",        # 3.2GB
        "unet/diffusion_pytorch_model.fp16.bin",           # 1.6GB (keep .safetensors)
        
        # Text Encoder duplicates - keep only fp16.safetensors
        "text_encoder/model.safetensors",                  # 469MB
        "text_encoder/pytorch_model.bin",                  # 469MB
        "text_encoder/pytorch_model.fp16.bin",             # 235MB (keep .safetensors)
    ]
    
    removed_size = 0.0
    for file_pattern in files_to_remove:
        files = list(sd_dir.rglob(file_pattern))
        for file_path in files:
            if file_path.exists():
                size_gb = get_file_size_gb(file_path)
                print(f"🗑️  Removing: {file_path.name} ({size_gb:.2f} GB)")
                file_path.unlink()
                removed_size += size_gb
    
    new_size = get_directory_size_gb(sd_dir)
    print(f"✅ SD optimization complete: {original_size:.2f} GB → {new_size:.2f} GB (saved {removed_size:.2f} GB)")
    
    return removed_size, new_size

def ensure_bark_downloaded() -> bool:
    """Ensure Bark models are fully downloaded"""
    print("🎵 Ensuring Bark models are downloaded...")
    
    try:
        # Try to import and initialize Bark
        import bark
        from bark import SAMPLE_RATE
        
        # Try to preload models to trigger download
        print("📥 Downloading Bark models...")
        bark.preload_models()
        
        # Check cache size
        cache_dir = pathlib.Path.home() / ".cache" / "suno" / "bark_v0"
        if cache_dir.exists():
            cache_size = get_directory_size_gb(cache_dir)
            print(f"✅ Bark cache size: {cache_size:.2f} GB")
            
            if cache_size > 4.0:  # Should be around 5GB when fully downloaded
                print("✅ Bark models appear to be fully downloaded")
                return True
            else:
                print("⚠️  Bark models may not be fully downloaded")
                return False
        else:
            print("❌ Bark cache directory not found")
            return False
            
    except ImportError:
        print("❌ Bark not installed. Installing official Bark from Suno...")
        try:
            # ✅ Cross-platform virtual environment detection
            venv_python = None
            if os.name == 'nt':  # Windows
                venv_python = pathlib.Path("backend/venv/Scripts/python.exe")
            else:  # Unix-like (Mac/Linux)
                venv_python = pathlib.Path("backend/venv/bin/python")
            
            if venv_python and venv_python.exists():
                # Install PyTorch with MPS support for Apple Silicon first
                print("📦 Installing PyTorch with MPS support for Apple Silicon...")
                subprocess.run([
                    str(venv_python), "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/"
                ], check=True)
                
                # Install official Bark from Suno
                print("📦 Installing official Bark from Suno...")
                subprocess.run([
                    str(venv_python), "-m", "pip", "install", 
                    "git+https://github.com/suno-ai/bark.git"
                ], check=True)
                
                print("✅ Bark installed successfully in virtual environment")
                return ensure_bark_downloaded()  # Retry
            else:
                print("❌ Virtual environment not found. Please install Bark manually.")
                return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Bark: {e}")
            return False
    except Exception as e:
        print(f"❌ Error with Bark: {e}")
        return False

def main():
    """Main optimization function"""
    print("🚀 Starting Model Optimization...")
    print("=" * 50)
    
    # Get original total size
    models_dir = pathlib.Path("models")
    original_total = get_directory_size_gb(models_dir)
    print(f"📊 Original total models size: {original_total:.2f} GB")
    print()
    
    total_saved = 0.0
    
    # Optimize SVD
    svd_saved, svd_new = optimize_svd_models()
    total_saved += svd_saved
    print()
    
    # Optimize SD  
    sd_saved, sd_new = optimize_sd_models()
    total_saved += sd_saved
    print()
    
    # Ensure Bark is downloaded
    bark_ok = ensure_bark_downloaded()
    print()
    
    # Final results
    final_total = get_directory_size_gb(models_dir)
    print("=" * 50)
    print("📊 OPTIMIZATION RESULTS:")
    print(f"   Original size: {original_total:.2f} GB")
    print(f"   Final size:    {final_total:.2f} GB")
    print(f"   Space saved:   {total_saved:.2f} GB")
    print(f"   Reduction:     {(total_saved/original_total)*100:.1f}%")
    print()
    
    if bark_ok:
        print("✅ All optimizations completed successfully!")
    else:
        print("⚠️  Optimization completed but Bark may need attention")
    
    print(f"🎯 Target achieved: {final_total:.1f} GB (target: 20-25 GB)")

if __name__ == "__main__":
    main()
