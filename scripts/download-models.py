#!/usr/bin/env python3
"""
Model Download Script for Text-to-Media App

This script downloads official model weights from Hugging Face and places them
in the local /models/ directory for offline use.

Usage:
    python scripts/download-models.py [--model MODEL_NAME] [--all]
    
Examples:
    python scripts/download-models.py --all                    # Download all models
    python scripts/download-models.py --model stable-video-diffusion
    python scripts/download-models.py --model bark
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

class ModelDownloader:
    """Downloads and manages model weights from Hugging Face"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations - Focused on the best local-first models
        self.video_models = {
            "stable-video-diffusion": {
                "repo": "stabilityai/stable-video-diffusion-img2vid",
                "path": "video/stable-video-diffusion",
                "description": "Stable Video Diffusion (SVD) - Official image-to-video generation",
                "size": "~5GB",
                "type": "img2vid",
                "resolution": "576×1024",
                "duration": "2-4 seconds",
                "files": [
                    "pytorch_model.bin",
                    "config.json",
                    "scheduler",
                    "transformer",
                    "vae"
                ]
            }
        }
        
        self.image_models = {
            "stable-diffusion": {
                "repo": "runwayml/stable-diffusion-v1-5",
                "path": "image/stable-diffusion",
                "description": "Stable Diffusion v1.5 - Text-to-image generation for SVD input",
                "size": "~4GB",
                "type": "text2img",
                "resolution": "512×512",
                "files": [
                    "pytorch_model.bin",
                    "config.json",
                    "scheduler",
                    "vae"
                ]
            }
        }
        
        self.audio_models = {
            "bark": {
                "repo": "suno/bark",
                "path": "audio/bark",
                "description": "Bark - High-quality text-to-speech and audio generation",
                "size": "~4GB",
                "type": "tts",
                "features": ["Multiple voices", "Non-speech sounds", "Offline capable"],
                "files": [
                    "pytorch_model.bin",
                    "config.json",
                    "tokenizer"
                ]
            }
        }
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        try:
            import huggingface_hub
            print("✅ huggingface_hub is installed")
            return True
        except ImportError:
            print("❌ huggingface_hub is not installed")
            print("Installing huggingface_hub...")
            try:
                # Try to use the virtual environment if it exists
                venv_python = Path(__file__).parent.parent / "backend" / "venv" / "bin" / "python"
                if venv_python.exists():
                    print(f"Using virtual environment: {venv_python}")
                    subprocess.check_call([str(venv_python), "-m", "pip", "install", "huggingface_hub"])
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "huggingface_hub"])
                print("✅ huggingface_hub installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install huggingface_hub")
                print("Please install it manually: pip install huggingface_hub")
                return False
    
    def download_model(self, model_type: str, model_name: str, force: bool = False) -> bool:
        """Download a specific model"""
        if model_type == "video" and model_name in self.video_models:
            model_config = self.video_models[model_name]
        elif model_type == "image" and model_name in self.image_models:
            model_config = self.image_models[model_name]
        elif model_type == "audio" and model_name in self.audio_models:
            model_config = self.audio_models[model_name]
        else:
            print(f"❌ Unknown model: {model_name}")
            return False
        
        model_path = self.models_dir / model_config["path"]
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        if model_path.exists() and any((model_path / file).exists() for file in model_config["files"]) and not force:
            print(f"✅ Model {model_name} already exists at {model_path}")
            return True
        
        print(f"📥 Downloading {model_name}...")
        print(f"   Repository: {model_config['repo']}")
        print(f"   Description: {model_config['description']}")
        print(f"   Size: {model_config['size']}")
        print(f"   Path: {model_path}")
        
        try:
            from huggingface_hub import snapshot_download
            
            # Download the model
            downloaded_path = snapshot_download(
                repo_id=model_config["repo"],
                local_dir=str(model_path),
                local_dir_use_symlinks=False,  # Use actual files, not symlinks
                resume_download=True
            )
            
            print(f"✅ Successfully downloaded {model_name} to {downloaded_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            return False
    
    def download_all_models(self, force: bool = False) -> Dict[str, bool]:
        """Download all available models"""
        results = {}
        
        print("🎬 Downloading video models...")
        for model_name in self.video_models:
            results[f"video_{model_name}"] = self.download_model("video", model_name, force)
        
        print("\n🖼️  Downloading image models...")
        for model_name in self.image_models:
            results[f"image_{model_name}"] = self.download_model("image", model_name, force)
        
        print("\n🎵 Downloading audio models...")
        for model_name in self.audio_models:
            results[f"audio_{model_name}"] = self.download_model("audio", model_name, force)
        
        return results
    
    def list_models(self):
        """List all available models"""
        print("📋 Available Models:")
        print("\n🎬 Video Models:")
        for name, config in self.video_models.items():
            status = "✅ Downloaded" if self._is_model_downloaded("video", name) else "❌ Not downloaded"
            print(f"   {name}: {config['description']} ({config['size']}) - {status}")
        
        print("\n🖼️  Image Models:")
        for name, config in self.image_models.items():
            status = "✅ Downloaded" if self._is_model_downloaded("image", name) else "❌ Not downloaded"
            print(f"   {name}: {config['description']} ({config['size']}) - {status}")
        
        print("\n🎵 Audio Models:")
        for name, config in self.audio_models.items():
            status = "✅ Downloaded" if self._is_model_downloaded("audio", name) else "❌ Not downloaded"
            print(f"   {name}: {config['description']} ({config['size']}) - {status}")
    
    def _is_model_downloaded(self, model_type: str, model_name: str) -> bool:
        """Check if a model is already downloaded"""
        if model_type == "video" and model_name in self.video_models:
            model_config = self.video_models[model_name]
        elif model_type == "image" and model_name in self.image_models:
            model_config = self.image_models[model_name]
        elif model_type == "audio" and model_name in self.audio_models:
            model_config = self.audio_models[model_name]
        else:
            return False
        
        model_path = self.models_dir / model_config["path"]
        return model_path.exists() and any((model_path / file).exists() for file in model_config["files"])
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model"""
        if model_name in self.video_models:
            return {"type": "video", **self.video_models[model_name]}
        elif model_name in self.image_models:
            return {"type": "image", **self.image_models[model_name]}
        elif model_name in self.audio_models:
            return {"type": "audio", **self.audio_models[model_name]}
        return None

def main():
    parser = argparse.ArgumentParser(description="Download model weights for Text-to-Media App")
    parser.add_argument("--model", help="Specific model to download")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--force", action="store_true", help="Force re-download even if model exists")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    # Check dependencies
    if not downloader.check_dependencies():
        sys.exit(1)
    
    if args.list:
        downloader.list_models()
        return
    
    if args.all:
        print("🚀 Downloading all models...")
        results = downloader.download_all_models(args.force)
        
        print("\n📊 Download Summary:")
        for model, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {model}: {status}")
        
        failed_count = sum(1 for success in results.values() if not success)
        if failed_count == 0:
            print("\n🎉 All models downloaded successfully!")
        else:
            print(f"\n⚠️  {failed_count} model(s) failed to download")
    
    elif args.model:
        model_info = downloader.get_model_info(args.model)
        if not model_info:
            print(f"❌ Unknown model: {args.model}")
            print("Available models:")
            downloader.list_models()
            sys.exit(1)
        
        success = downloader.download_model(model_info["type"], args.model, args.force)
        if success:
            print(f"🎉 Model {args.model} downloaded successfully!")
        else:
            print(f"❌ Failed to download model {args.model}")
            sys.exit(1)
    
    else:
        print("❓ Please specify --model MODEL_NAME, --all, or --list")
        print("\nAvailable models:")
        downloader.list_models()

if __name__ == "__main__":
    main()
