#!/usr/bin/env python3
"""
Comprehensive Model Download Script

This script downloads all the recommended models for the text-to-media application:
- AnimateDiff (for GIF generation)
- Stable Diffusion (for text-to-image generation)
- Bark (for text-to-speech)
"""

import os
import sys
import argparse
import pathlib
import shutil
import time
import requests
from typing import Dict, List, Optional
from huggingface_hub import snapshot_download
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "stable_diffusion": {
        "name": "Stable Diffusion v1.5",
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "local_dir": "../models/image/stable-diffusion",
        "size_gb": 44.0,
        "priority": 1,  # High priority - base model
        "description": "Text-to-image generation (base model for AnimateDiff)"
    },
    "animatediff": {
        "name": "AnimateDiff Official Repository",
        "repo_id": "guoyww/AnimateDiff",
        "local_dir": "../models/video/animatediff",
        "size_gb": 5.0,
        "priority": 3,
        "description": "Official AnimateDiff repository with motion adapters and configs"
    },
    "animatediff_motion_adapter": {
        "name": "AnimateDiff Motion Adapter v1.5.2",
        "repo_id": "guoyww/animatediff-motion-adapter-v1-5-2",
        "local_dir": "../models/video/animatediff/motion_adapter",
        "size_gb": 2.0,
        "priority": 3,
        "description": "Motion adapter for AnimateDiff v1.5.2"
    },
    "bark": {
        "name": "Bark",
        "repo_id": "suno/bark",
        "local_dir": "../models/audio/bark",
        "size_gb": 5.0,
        "priority": 2,  # High priority - primary TTS model
        "description": "Text-to-speech generation"
    }
}

def check_existing_models() -> Dict[str, bool]:
    """Check which models are already downloaded"""
    existing = {}
    
    for model_id, config in MODELS.items():
        local_dir = pathlib.Path(config["local_dir"])
        
        if local_dir.exists():
            # Check for actual model weight files
            weight_files = list(local_dir.rglob("*.safetensors")) + list(local_dir.rglob("*.bin")) + list(local_dir.rglob("*.pt")) + list(local_dir.rglob("*.pth"))
            
            if len(weight_files) > 0:
                existing[model_id] = True
                logger.info(f"✅ {config['name']} already downloaded")
            else:
                existing[model_id] = False
                logger.warning(f"⚠️  {config['name']} directory exists but no model files found")
        else:
            existing[model_id] = False
            logger.info(f"❌ {config['name']} not found")
    
    return existing

def check_network_connectivity() -> bool:
    """Check if we can reach Hugging Face"""
    try:
        response = requests.get("https://huggingface.co", timeout=10)
        return response.status_code == 200
    except Exception:
        return False

def download_model_with_retry(model_id: str, force: bool = False, max_retries: int = 3) -> bool:
    """Download a specific model with retry logic and network resilience"""
    if model_id not in MODELS:
        logger.error(f"Unknown model: {model_id}")
        return False

    config = MODELS[model_id]
    local_dir = pathlib.Path(config["local_dir"])

    # Skip if already downloaded (weights exist) unless force
    if not force and local_dir.exists():
        weight_files = (
            list(local_dir.rglob("*.safetensors"))
            + list(local_dir.rglob("*.bin"))
            + list(local_dir.rglob("*.pt"))
            + list(local_dir.rglob("*.pth"))
        )
        if len(weight_files) > 0:
            logger.info(f"⏭️  Skipping {config['name']} - already downloaded")
            return True

    # Check network connectivity first
    if not check_network_connectivity():
        logger.warning("⚠️  Network connectivity issues detected. Will retry with longer timeouts...")

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"🔄 Retry attempt {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)

            logger.info(f"📥 Downloading {config['name']}...")
            logger.info(f"   Repository: {config['repo_id']}")
            logger.info(f"   Local path: {local_dir}")
            logger.info(f"   Size: ~{config['size_gb']}GB")
            logger.info(f"   Attempt: {attempt + 1}/{max_retries}")

            # Ensure directory
            local_dir.mkdir(parents=True, exist_ok=True)

            # Enhanced download with better error handling
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=str(local_dir),
                force_download=bool(force),
                resume_download=True,
                local_dir_use_symlinks=False,
                max_workers=1,  # Reduce concurrent downloads to avoid network issues
                tqdm_class=None,  # Disable progress bar to reduce output noise
            )

            # Verify: require at least one weight file
            weight_files = (
                list(local_dir.rglob("*.safetensors"))
                + list(local_dir.rglob("*.bin"))
                + list(local_dir.rglob("*.pt"))
                + list(local_dir.rglob("*.pth"))
            )
            
            # Also check for config files to ensure complete download
            config_files = list(local_dir.rglob("*.json")) + list(local_dir.rglob("*.txt")) + list(local_dir.rglob("*.yaml"))
            
            if len(weight_files) == 0:
                raise Exception("No weight files (*.safetensors|*.bin|*.pt|*.pth) found after download")
            
            if len(config_files) == 0:
                logger.warning("⚠️  No config files found - model may be incomplete")

            total_size = sum(f.stat().st_size for f in weight_files if f.is_file())
            size_gb = total_size / (1024 * 1024 * 1024)
            logger.info(f"✅ Successfully downloaded {config['name']} ({size_gb:.2f} GB of weights)")
            
            # Log all downloaded files for verification
            all_files = list(local_dir.rglob("*"))
            logger.info(f"📁 Downloaded {len(all_files)} files total")
            
            return True

        except Exception as e:
            logger.error(f"❌ Attempt {attempt + 1} failed for {config['name']}: {e}")
            
            if attempt == max_retries - 1:
                logger.error(f"💥 All {max_retries} attempts failed for {config['name']}")
                return False
            else:
                logger.info(f"🔄 Will retry in {2 ** (attempt + 1)} seconds...")

    return False

def download_model(model_id: str, force: bool = False) -> bool:
    """Download a specific model (wrapper for backward compatibility)"""
    return download_model_with_retry(model_id, force, max_retries=3)
    
def download_bark_preset_audios() -> bool:
    """Download Bark preset audio files for voice previews"""
    try:
        logger.info("🎵 Downloading Bark preset audio files...")
        
        # Create preset audio directory
        preset_dir = pathlib.Path("../models/audio/bark/preset-audios")
        preset_dir.mkdir(parents=True, exist_ok=True)
        
        # English voice presets with their audio URLs
        preset_audios = [
            {
                "id": "v2/en_speaker_0",
                "name": "Speaker 0 (EN)",
                "prompt_url": "https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_0.mp3",
                "continuation_url": "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_0.mp3"
            },
            {
                "id": "v2/en_speaker_1", 
                "name": "Speaker 1 (EN)",
                "prompt_url": "https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_1.mp3",
                "continuation_url": "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_1.mp3"
            },
            {
                "id": "v2/en_speaker_2",
                "name": "Speaker 2 (EN)",
                "prompt_url": "https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_2.mp3",
                "continuation_url": "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_2.mp3"
            },
            {
                "id": "v2/en_speaker_3",
                "name": "Speaker 3 (EN)",
                "prompt_url": "https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_3.mp3",
                "continuation_url": "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_3.mp3"
            },
            {
                "id": "v2/en_speaker_4",
                "name": "Speaker 4 (EN)",
                "prompt_url": "https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_4.mp3",
                "continuation_url": "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_4.mp3"
            },
            {
                "id": "v2/en_speaker_5",
                "name": "Speaker 5 (EN)",
                "prompt_url": "https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_5.mp3",
                "continuation_url": "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_5.mp3"
            },
            {
                "id": "v2/en_speaker_6",
                "name": "Speaker 6 (EN)",
                "prompt_url": "https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_6.mp3",
                "continuation_url": "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_6.mp3"
            },
            {
                "id": "v2/en_speaker_7",
                "name": "Speaker 7 (EN)",
                "prompt_url": "https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_7.mp3",
                "continuation_url": "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_7.mp3"
            },
            {
                "id": "v2/en_speaker_8",
                "name": "Speaker 8 (EN)",
                "prompt_url": "https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_8.mp3",
                "continuation_url": "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_8.mp3"
            },
            {
                "id": "v2/en_speaker_9",
                "name": "Speaker 9 (EN)",
                "prompt_url": "https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_9.mp3",
                "continuation_url": "https://dl.suno-models.io/bark/prompts/continuation_audio/en_speaker_9.mp3"
            }
        ]
        
        downloaded_count = 0
        for preset in preset_audios:
            try:
                # Download prompt audio (main preview)
                prompt_file = preset_dir / f"{preset['id'].replace('/', '_')}-preview.mp3"
                if not prompt_file.exists():
                    logger.info(f"📥 Downloading {preset['name']} prompt audio...")
                    response = requests.get(preset['prompt_url'], timeout=30)
                    response.raise_for_status()
                    with open(prompt_file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✅ Downloaded {preset['name']} prompt audio")
                else:
                    logger.info(f"⏭️  {preset['name']} prompt audio already exists")
                
                # Download continuation audio (optional)
                continuation_file = preset_dir / f"{preset['id'].replace('/', '_')}-continuation.mp3"
                if not continuation_file.exists():
                    logger.info(f"📥 Downloading {preset['name']} continuation audio...")
                    response = requests.get(preset['continuation_url'], timeout=30)
                    response.raise_for_status()
                    with open(continuation_file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✅ Downloaded {preset['name']} continuation audio")
                else:
                    logger.info(f"⏭️  {preset['name']} continuation audio already exists")
                
                downloaded_count += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to download {preset['name']}: {e}")
                continue
        
        if downloaded_count > 0:
            logger.info(f"✅ Downloaded {downloaded_count} Bark preset audio sets")
            return True
        else:
            logger.warning("⚠️  No Bark preset audios were downloaded")
            return False
            
    except Exception as e:
        logger.error(f"❌ Bark preset audio download failed: {e}")
        return False

def download_bark_models() -> bool:
    """Special handling for Bark models"""
    try:
        logger.info("🎵 Setting up Bark models...")
        
        # Try to import and preload Bark
        try:
            import bark
            from bark import preload_models
            
            # Preload models (this downloads them to cache)
            preload_models()
            
            # Check if Bark cache exists
            bark_cache = pathlib.Path.home() / ".cache" / "suno" / "bark_v0"
            if bark_cache.exists():
                # Calculate cache size
                total_size = sum(f.stat().st_size for f in bark_cache.rglob("*") if f.is_file())
                size_mb = total_size / (1024 * 1024)
                
                logger.info(f"✅ Bark models ready ({size_mb:.1f} MB)")
                
                # Download preset audios after successful model setup
                download_bark_preset_audios()
                
                return True
            else:
                raise Exception("Bark cache not found after preload")
                
        except ImportError:
            # Fallback: download Bark repository (will include configs)
            logger.info("📦 Bark package not found, downloading repository...")
            success = download_model("bark")
            if success:
                # Download preset audios after successful model download
                download_bark_preset_audios()
            return success
            
    except Exception as e:
        logger.error(f"❌ Bark setup failed: {e}")
        return False
    
def download_animatediff_models() -> bool:
    """Special handling for AnimateDiff models following official repository structure"""
    try:
        logger.info("🎬 Setting up AnimateDiff models...")
        
        # Download the official AnimateDiff repository
        logger.info("📥 Downloading AnimateDiff official repository...")
        repo_success = download_model("animatediff")
        if not repo_success:
            logger.error("❌ Failed to download AnimateDiff repository")
            return False
        
        # Download the motion adapter separately
        logger.info("📥 Downloading AnimateDiff motion adapter...")
        adapter_success = download_model("animatediff_motion_adapter")
        if not adapter_success:
            logger.error("❌ Failed to download AnimateDiff motion adapter")
            return False
        
        # Verify the setup
        animatediff_dir = pathlib.Path("../models/video/animatediff")
        motion_adapter_dir = pathlib.Path("../models/video/animatediff/motion_adapter")
        
        # Check for essential files
        essential_files = [
            "configs/prompts/1_animate/1_1_animate_RealisticVision.yaml",
            "scripts/animate.py",
            "app.py"
        ]
        
        missing_files = []
        for file_path in essential_files:
            if not (animatediff_dir / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"⚠️  Missing essential files: {missing_files}")
            return False
        
        # Check motion adapter files
        adapter_files = list(motion_adapter_dir.rglob("*.safetensors")) + list(motion_adapter_dir.rglob("*.bin"))
        if len(adapter_files) == 0:
            logger.error("❌ No motion adapter weight files found")
            return False
        
        logger.info("✅ AnimateDiff setup completed successfully")
        logger.info(f"📁 Repository: {animatediff_dir}")
        logger.info(f"📁 Motion adapter: {motion_adapter_dir}")
        logger.info(f"📁 Config files: {len(list(animatediff_dir.rglob('*.yaml')))} YAML configs found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AnimateDiff setup failed: {e}")
        return False

        
def verify_model_integrity(model_id: str) -> bool:
    """Verify that a model download is complete and not corrupted"""
    if model_id not in MODELS:
        return False
    
    config = MODELS[model_id]
    local_dir = pathlib.Path(config["local_dir"])
    
    if not local_dir.exists():
        return False
    
    # Check for essential files
    weight_files = (
        list(local_dir.rglob("*.safetensors"))
        + list(local_dir.rglob("*.bin"))
        + list(local_dir.rglob("*.pt"))
        + list(local_dir.rglob("*.pth"))
    )
    
    config_files = (
        list(local_dir.rglob("*.json"))
        + list(local_dir.rglob("*.txt"))
        + list(local_dir.rglob("*.yaml"))
        + list(local_dir.rglob("*.yml"))
    )
    
    # Must have at least one weight file
    if len(weight_files) == 0:
        logger.warning(f"⚠️  {config['name']}: No weight files found")
        return False
    
    # Check for file corruption (basic size check)
    corrupted_files = []
    for weight_file in weight_files:
        if weight_file.stat().st_size == 0:
            corrupted_files.append(weight_file.name)
    
    if corrupted_files:
        logger.warning(f"⚠️  {config['name']}: Corrupted files detected: {corrupted_files}")
        return False
    
    # Log verification results
    total_size = sum(f.stat().st_size for f in weight_files if f.is_file())
    size_gb = total_size / (1024 * 1024 * 1024)
    
    logger.info(f"✅ {config['name']}: Verified ({len(weight_files)} weight files, {len(config_files)} config files, {size_gb:.2f}GB)")
    return True

def download_all_models(force: bool = False, priority_only: bool = False) -> bool:
    """Download all models in priority order with comprehensive verification"""
    logger.info("🚀 Starting comprehensive model download...")
    
    # Check network connectivity
    if not check_network_connectivity():
        logger.warning("⚠️  Network connectivity issues detected. Downloads may fail or be slow.")
    
    # Check existing models
    existing = check_existing_models()
    
    # Sort models by priority
    sorted_models = sorted(MODELS.items(), key=lambda x: x[1]["priority"])
    
    if priority_only:
        # Only download high-priority models (1-3)
        sorted_models = [m for m in sorted_models if m[1]["priority"] <= 3]
        logger.info("📋 Downloading priority models only (AnimateDiff, Kandinsky, XTTS-v2, SD)")
    
    total_models = len(sorted_models)
    completed = 0
    failed = []
    
    for model_id, config in sorted_models:
        try:
            logger.info(f"📊 Progress: {completed}/{total_models} models completed")
            
            # Skip if already downloaded and verified (unless force)
            if not force and existing.get(model_id, False):
                if verify_model_integrity(model_id):
                    logger.info(f"⏭️  Skipping {config['name']} - already downloaded and verified")
                    completed += 1
                    continue
            
            # Download the model
            if model_id == "bark":
                success = download_bark_models()
            elif model_id == "xtts-v2":
                success = download_xtts_models()
            elif model_id == "animatediff":
                success = download_animatediff_models()
            elif model_id == "animatediff_motion_adapter":
                # Skip motion adapter as it's handled by download_animatediff_models
                success = True
            else:
                success = download_model_with_retry(model_id, force, max_retries=5)  # More retries for critical models
            
            if success:
                # Verify the download
                if verify_model_integrity(model_id):
                    completed += 1
                    logger.info(f"✅ {config['name']} downloaded and verified successfully")
                else:
                    logger.error(f"❌ {config['name']} download verification failed")
                    failed.append(model_id)
            else:
                failed.append(model_id)
                
        except Exception as e:
            logger.error(f"❌ Error downloading {config['name']}: {e}")
            failed.append(model_id)
    
    # Final verification of all models
    logger.info("🔍 Performing final verification of all models...")
    verification_failed = []
    for model_id, config in sorted_models:
        if not verify_model_integrity(model_id):
            verification_failed.append(model_id)
    
    # Summary
    logger.info(f"🎉 Download completed: {completed}/{total_models} models successful")
    
    if failed:
        logger.warning(f"⚠️  Failed downloads: {', '.join(failed)}")
    
    if verification_failed:
        logger.warning(f"⚠️  Verification failed: {', '.join(verification_failed)}")
        return False
    
    if failed:
        logger.warning("⚠️  Some downloads failed, but existing models are verified")
        return False
    else:
        logger.info("✅ All models downloaded and verified successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Download AI models for text-to-media application")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--priority", action="store_true", help="Download priority models only (AnimateDiff, Kandinsky, XTTS-v2, SD)")
    parser.add_argument("--model", type=str, help="Download specific model")
    parser.add_argument("--force", action="store_true", help="Force re-download even if model exists")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--verify", action="store_true", help="Verify existing model downloads")
    parser.add_argument("--retries", type=int, default=3, help="Number of retry attempts for failed downloads")
    
    args = parser.parse_args()
    
    if args.list:
        logger.info("📋 Available models:")
        for model_id, config in sorted(MODELS.items(), key=lambda x: x[1]["priority"]):
            priority_text = "🔥" if config["priority"] <= 3 else "📦"
            status = "✅" if verify_model_integrity(model_id) else "❌"
            logger.info(f"  {priority_text} {status} {model_id}: {config['name']} (~{config['size_gb']}GB)")
            logger.info(f"     {config['description']}")
        return
    
    if args.verify:
        logger.info("🔍 Verifying all model downloads...")
        all_good = True
        for model_id, config in MODELS.items():
            if verify_model_integrity(model_id):
                logger.info(f"✅ {config['name']}: OK")
            else:
                logger.error(f"❌ {config['name']}: FAILED")
                all_good = False
        
        if all_good:
            logger.info("🎉 All models verified successfully!")
        else:
            logger.error("⚠️  Some models failed verification!")
            sys.exit(1)
        return
    
    if args.model:
        if args.model not in MODELS:
            logger.error(f"Unknown model: {args.model}")
            logger.info("Available models:", list(MODELS.keys()))
            return
        
        success = download_model_with_retry(args.model, args.force, max_retries=args.retries)
        if success and verify_model_integrity(args.model):
            logger.info("✅ Model download and verification completed successfully!")
        else:
            logger.error("❌ Model download or verification failed!")
            sys.exit(1)
        
    elif args.all or args.priority:
        success = download_all_models(args.force, args.priority)
        if success:
            logger.info("✅ All model downloads completed successfully!")
        else:
            logger.error("❌ Some model downloads failed!")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()