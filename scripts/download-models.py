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
                "size": "~5GB",
                "type": "tts",
                "features": ["Multiple voices", "Non-speech sounds", "Offline capable"],
                "files": [
                    "pytorch_model.bin",
                    "config.json",
                    "tokenizer",
                    "scheduler",
                    "vae",
                    "transformer",
                    "text_2.pt",
                    "text_2.pt.safetensors",
                    "coarse_2.pt",
                    "coarse_2.pt.safetensors",
                    "fine_2.pt",
                    "fine_2.pt.safetensors"
                ],
                "voices": [
                    "v2/en_speaker_0",  # Default English speaker
                    "v2/en_speaker_1",  # Alternative English speaker
                    "v2/en_speaker_2",  # Another English speaker
                    "v2/en_speaker_3",  # Another English speaker
                    "v2/en_speaker_4",  # Another English speaker
                    "v2/en_speaker_5",  # Another English speaker
                    "v2/en_speaker_6",  # Another English speaker
                    "v2/en_speaker_7",  # Another English speaker
                    "v2/en_speaker_8",  # Another English speaker
                    "v2/en_speaker_9",  # Another English speaker
                    "v2/en_speaker_6",  # Female voice
                    "v2/en_speaker_7",  # Male voice
                    "v2/en_speaker_8",  # Child voice
                    "v2/en_speaker_9",  # Narrator voice
                ]
            }
        }
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        required_packages = ["huggingface_hub", "torch", "transformers"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} is not installed")
        
        if missing_packages:
            print(f"Installing missing packages: {', '.join(missing_packages)}...")
            try:
                # ✅ Cross-platform virtual environment detection
                venv_python = None
                if os.name == 'nt':  # Windows
                    venv_python = Path(__file__).parent.parent / "backend" / "venv" / "Scripts" / "python.exe"
                else:  # Unix-like (Mac/Linux)
                    venv_python = Path(__file__).parent.parent / "backend" / "venv" / "bin" / "python"
                
                if venv_python and venv_python.exists():
                    print(f"Using virtual environment: {venv_python}")
                    for package in missing_packages:
                        subprocess.check_call([str(venv_python), "-m", "pip", "install", package])
                else:
                    print("Virtual environment not found, installing to user directory...")
                    for package in missing_packages:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
                print("✅ All required packages installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install required packages")
                print(f"Please install them manually: pip install {' '.join(missing_packages)}")
                return False
        
        return True
    
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
        
        # Check if model already exists and has actual weight files
        has_weights = False
        if model_path.exists():
            # Look for actual model weight files
            weight_files = list(model_path.rglob("*.safetensors")) + list(model_path.rglob("*.bin")) + list(model_path.rglob("*.pt")) + list(model_path.rglob("*.pth"))
            has_weights = len(weight_files) > 0
        
        if has_weights and not force:
            print(f"✅ Model {model_name} already exists at {model_path}")
            return True
        
        print(f"📥 Downloading {model_name}...")
        print(f"   Repository: {model_config['repo']}")
        print(f"   Description: {model_config['description']}")
        print(f"   Size: {model_config['size']}")
        print(f"   Path: {model_path}")
        
        try:
            from huggingface_hub import snapshot_download
            from tqdm.auto import tqdm
            
            # Special handling for Bark - use preload_models() to download all files
            if model_name == "bark":
                return self._download_bark_model(model_path)
            else:
                # Download the model with progress bar
                print(f"📥 Downloading {model_name} from {model_config['repo']}...")
                downloaded_path = snapshot_download(
                    repo_id=model_config["repo"],
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,  # Use actual files, not symlinks
                    resume_download=True,
                    tqdm_class=tqdm  # ✅ Shows download progress bar
                )
            
            print(f"✅ Successfully downloaded {model_name} to {downloaded_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            return False
    
    def _download_bark_model(self, model_path: Path) -> bool:
        """Download Bark model using preload_models() to get all required files and voices"""
        try:
            print("🔧 Installing Bark package...")
            try:
                import bark
                print("✅ Bark package already installed")
            except ImportError:
                print("📦 Installing Bark package from GitHub...")
                # ✅ Cross-platform virtual environment detection
                venv_python = None
                if os.name == 'nt':  # Windows
                    venv_python = Path(__file__).parent.parent / "backend" / "venv" / "Scripts" / "python.exe"
                else:  # Unix-like (Mac/Linux)
                    venv_python = Path(__file__).parent.parent / "backend" / "venv" / "bin" / "python"
                
                if venv_python and venv_python.exists():
                    subprocess.check_call([str(venv_python), "-m", "pip", "install", "git+https://github.com/suno-ai/bark.git"])
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/suno-ai/bark.git"])
                print("✅ Bark package installed successfully")
            
            print("📥 Preloading Bark models (this downloads all required files)...")
            import bark
            bark.preload_models()
            
            # ✅ Extra: Ensure voices are cached locally
            print("🎤 Downloading voice prompts for offline use...")
            try:
                from bark.generation import download_model
                bark_config = self.audio_models["bark"]
                if "voices" in bark_config:
                    for voice in bark_config["voices"]:
                        try:
                            print(f"   Downloading voice: {voice}")
                            download_model(voice)
                        except Exception as voice_error:
                            print(f"⚠️  Could not download voice {voice}: {voice_error}")
                            print("   Continuing with other voices...")
            except ImportError:
                print("⚠️  Could not import bark.generation.download_model, skipping voice downloads")
            except Exception as e:
                print(f"⚠️  Error downloading voices: {e}")
            
            print("✅ Successfully downloaded all Bark model files & voices")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download Bark model: {e}")
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
        
        # Generate voice previews after downloading Bark
        if "bark" in self.audio_models and results.get("audio_bark", False):
            print("\n🎤 Generating voice previews...")
            self.generate_voice_previews()
        
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
        """Check if a model is already downloaded with all required files"""
        if model_type == "video" and model_name in self.video_models:
            model_config = self.video_models[model_name]
        elif model_type == "image" and model_name in self.image_models:
            model_config = self.image_models[model_name]
        elif model_type == "audio" and model_name in self.audio_models:
            model_config = self.audio_models[model_name]
        else:
            return False
        
        model_path = self.models_dir / model_config["path"]
        if not model_path.exists():
            return False
        
        # ✅ Ensure ALL required files exist (not just any one file)
        for required_file in model_config["files"]:
            # Use rglob to find files that match the pattern (handles subdirectories)
            matching_files = list(model_path.rglob(required_file))
            if not matching_files:
                print(f"⚠️  Missing required file: {required_file} for {model_name}")
                return False
        
        print(f"✅ All required files present for {model_name}")
        return True
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model"""
        if model_name in self.video_models:
            model_config = self.video_models[model_name].copy()
            model_config["type"] = "video"  # Override the internal type
            return model_config
        elif model_name in self.image_models:
            model_config = self.image_models[model_name].copy()
            model_config["type"] = "image"  # Override the internal type
            return model_config
        elif model_name in self.audio_models:
            model_config = self.audio_models[model_name].copy()
            model_config["type"] = "audio"  # Override the internal type
            return model_config
        return None
    
    def generate_voice_previews(self):
        """Generate voice preview samples for each Bark voice"""
        try:
            print("🎤 Generating voice preview samples...")
            
            # Create voice previews directory
            previews_dir = self.base_dir / "outputs" / "voice-previews"
            previews_dir.mkdir(parents=True, exist_ok=True)
            
            # Sample texts for different voice types
            voice_samples = {
                "v2/en_speaker_0": "Hello! This is the default English speaker voice. Clear and natural.",
                "v2/en_speaker_1": "Greetings! I'm an alternative English speaker with a distinct tone.",
                "v2/en_speaker_2": "Hi there! This voice has a warm and friendly character.",
                "v2/en_speaker_3": "Good day! This speaker has a professional and authoritative tone.",
                "v2/en_speaker_4": "Hello! This voice is perfect for storytelling and narration.",
                "v2/en_speaker_5": "Hey! This speaker has a casual and conversational style.",
                "v2/en_speaker_6": "Hi! This is a female voice that's clear and engaging.",
                "v2/en_speaker_7": "Hello! This is a male voice with depth and character.",
                "v2/en_speaker_8": "Hi! This is a young, energetic voice perfect for children's content.",
                "v2/en_speaker_9": "Welcome. This is a narrator voice, clear and authoritative."
            }
            
            # Try to import Bark and generate previews
            try:
                import bark
                from bark import generate_audio, SAMPLE_RATE
                import soundfile as sf
                import numpy as np
                
                print("✅ Bark imported successfully, generating voice previews...")
                
                for voice_id, sample_text in voice_samples.items():
                    try:
                        print(f"🎵 Generating preview for {voice_id}...")
                        
                        # Generate audio with specific voice
                        audio_array = generate_audio(sample_text, history_prompt=voice_id)
                        
                        # Save preview file
                        preview_filename = f"{voice_id.replace('/', '_')}-preview.wav"
                        preview_path = previews_dir / preview_filename
                        
                        # Ensure audio is in the right format
                        if isinstance(audio_array, np.ndarray):
                            sf.write(str(preview_path), audio_array, SAMPLE_RATE)
                            print(f"✅ Generated preview: {preview_filename}")
                        else:
                            print(f"⚠️  Skipped {voice_id} - invalid audio format")
                            
                    except Exception as e:
                        print(f"⚠️  Could not generate preview for {voice_id}: {e}")
                        continue
                
                print("🎉 Voice preview generation completed!")
                
            except ImportError:
                print("⚠️  Bark not available for voice preview generation")
                print("   Voice previews will be generated when Bark is properly installed")
                
        except Exception as e:
            print(f"❌ Error generating voice previews: {e}")

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
