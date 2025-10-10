from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
import os
import sys
import json
from typing import List, Optional
from pydantic import BaseModel
import asyncio
from datetime import datetime
import time
import uuid
import pathlib
import queue
import subprocess
import threading

from models.ultimate_video_generator import UltimateVideoGenerator
from models.audio_generator import AudioGenerator
from models.image_generator import ImageGenerator
from utils.gpu_detector import GPUDetector
from utils.ffmpeg_handler import FFmpegHandler
import shutil
from utils.job_queue import JobQueue
import subprocess
import threading
import time
import os
from huggingface_hub import snapshot_download
from config import get_config

app = FastAPI(title="Text-to-Media Generator", version="1.0.0")

# Initialize configuration
config = get_config()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounting will be done at the end after all routes are defined
# Note: We'll use FileResponse for binary files instead of StaticFiles to avoid encoding issues

# Mount static files for models directory
app.mount("/models", StaticFiles(directory="../models"), name="models")

# Initialize components
gpu_detector = GPUDetector()
ffmpeg_handler = FFmpegHandler()
job_queue = JobQueue()

# Global model instances
video_generator = None
audio_generator = None
image_generator = None

# Unified model download status tracking
download_status = {
    "is_downloading": False,
    "overall_progress": 0,
    "current_model": "",
    "status": "idle",  # idle, downloading, completed, error
    "message": "",
    "error": None,
    "models": {
        "stable_diffusion": {
            "name": "Stable Diffusion",
            "repo_id": "runwayml/stable-diffusion-v1-5",
            "local_dir": "../models/image/stable-diffusion",
            "size_gb": 44.0,
            "status": "pending",  # pending, downloading, done, error
            "progress": 0,
            "downloaded_mb": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
            "files_verified": False
        },
        "svd": {
            "name": "Stable Video Diffusion",
            "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "local_dir": "../models/video/svd",
            "size_gb": 5.0,
            "status": "pending",
            "progress": 0,
            "downloaded_mb": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
            "files_verified": False
        },
        "bark": {
            "name": "Bark",
            "repo_id": "suno/bark",
            "local_dir": "../models/audio/bark",
            "size_gb": 5.0,
            "status": "pending",
            "progress": 0,
            "downloaded_mb": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
            "files_verified": False
        }
    }
}

# Track running per-model downloads so we can cancel them
running_downloads: dict[str, subprocess.Popen] = {}
cancel_flags: dict[str, bool] = {}

class GenerationRequest(BaseModel):
    prompt: str
    model_type: str  # "video" or "audio" or "image"
    model_name: str
    duration: Optional[int] = 5  # for video
    sample_rate: Optional[int] = 22050  # for audio
    output_format: str = "mp4"  # mp4, wav, mp3
    voice_style: Optional[str] = "auto"  # for audio voice selection
    voice_id: Optional[str] = None  # specific Bark voice ID
    # Advanced video settings
    resolution: Optional[int] = 512  # 256, 384, or 512
    num_frames: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    motion_scale: Optional[float] = None
    fps: Optional[int] = None
    seed: Optional[int] = None

class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    output_file: Optional[str] = None
    error: Optional[str] = None

def check_existing_models():
    """Check which models are already downloaded and update status"""
    global download_status
    
    print("🔍 Checking existing models...")
    
    for model_key, model_info in download_status["models"].items():
        local_dir = pathlib.Path(model_info["local_dir"])
        
        # Special-case: Bark stores weights in user cache (~/.cache/suno/bark_v0)
        if model_key == "bark":
            bark_cache = pathlib.Path.home() / ".cache" / "suno" / "bark_v0"
            local_bark_dir = pathlib.Path(model_info["local_dir"])
            
            # Check both cache and local directory
            bark_available = False
            
            # Check cache first
            if bark_cache.exists():
                weight_files = list(bark_cache.rglob("*.pt")) + list(bark_cache.rglob("*.pth")) + list(bark_cache.rglob("*.bin")) + list(bark_cache.rglob("*.safetensors"))
                if len(weight_files) > 0:
                    total_size = sum(f.stat().st_size for f in weight_files if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    download_status["models"][model_key].update({
                        "status": "done",
                        "progress": 100,
                        "downloaded_mb": size_mb,
                        "files_verified": True
                    })
                    print(f"✅ Bark models found in cache")
                    bark_available = True
            
            # Also check local directory for complete Bark setup
            if local_bark_dir.exists():
                # Check for English-only preset audios and speaker embeddings
                preset_audios_dir = local_bark_dir / "preset-audios"
                speaker_embeddings_dir = local_bark_dir / "speaker_embeddings"
                
                english_presets = list(preset_audios_dir.glob("en_speaker_*.npz")) if preset_audios_dir.exists() else []
                english_embeddings = list(speaker_embeddings_dir.glob("en_speaker_*_*.npy")) if speaker_embeddings_dir.exists() else []
                
                if len(english_presets) >= 10 and len(english_embeddings) >= 20:
                    print(f"✅ Bark local setup complete (English only: {len(english_presets)} presets, {len(english_embeddings)} embeddings)")
                    bark_available = True
            
            if bark_available:
                continue

        if local_dir.exists():
            # Check for actual model weight files
            weight_files = list(local_dir.rglob("*.safetensors")) + list(local_dir.rglob("*.bin")) + list(local_dir.rglob("*.pt")) + list(local_dir.rglob("*.pth"))
            
            # Special check for SVD - ensure we have all required model files
            if model_key == "svd":
                # Check for specific SVD model files
                required_files = ["model_index.json", "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"]
                has_required = all((local_dir / file).exists() for file in required_files)
                
                if len(weight_files) > 0 and has_required:
                    # Calculate total size
                    total_size = sum(f.stat().st_size for f in weight_files if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    
                    download_status["models"][model_key].update({
                        "status": "done",
                        "progress": 100,
                        "downloaded_mb": size_mb,
                        "files_verified": True
                    })
                    print(f"✅ {model_info['name']} already downloaded")
                else:
                    # Missing required files
                    download_status["models"][model_key].update({
                        "status": "pending",
                        "progress": 0,
                        "downloaded_mb": 0,
                        "files_verified": False
                    })
                    print(f"⚠️  {model_info['name']} found but required files missing - will re-download")
            elif len(weight_files) > 0:
                # Calculate total size
                total_size = sum(f.stat().st_size for f in weight_files if f.is_file())
                size_mb = total_size / (1024 * 1024)
                
                download_status["models"][model_key].update({
                    "status": "done",
                    "progress": 100,
                    "downloaded_mb": size_mb,
                    "files_verified": True
                })
                print(f"✅ {model_info['name']} already downloaded")
            else:
                # Directory exists but no model files - mark as pending
                download_status["models"][model_key].update({
                    "status": "pending",
                    "progress": 0,
                    "downloaded_mb": 0,
                    "files_verified": False
                })
                print(f"⚠️  {model_info['name']} directory exists but no model files found - will re-download")
        else:
            # Directory doesn't exist - mark as pending
            download_status["models"][model_key].update({
                "status": "pending",
                "progress": 0,
                "downloaded_mb": 0,
                "files_verified": False
            })
            print(f"❌ {model_info['name']} not found - will download")
    
    # Update overall progress
    completed_models = sum(1 for model in download_status["models"].values() if model["status"] == "done")
    total_models = len(download_status["models"])
    download_status["overall_progress"] = int((completed_models / total_models) * 100)
    
    if completed_models == total_models:
        download_status["status"] = "completed"
        print("🎉 All models are already downloaded!")
    else:
        print(f"📊 {completed_models}/{total_models} models downloaded ({download_status['overall_progress']}%)")

def safe_snapshot_download(model_id: str, local_dir: str, model_key: str):
    """Safely download a model with progress tracking"""
    global download_status
    
    try:
        print(f"📥 Starting download of {model_id}...")
        download_status["models"][model_key]["status"] = "downloading"
        download_status["current_model"] = model_key
        download_status["message"] = f"Downloading {download_status['models'][model_key]['name']}..."
        
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download with resume capability
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        
        # Verify download
        local_path = pathlib.Path(local_dir)
        if not local_path.exists() or len(list(local_path.iterdir())) == 0:
            raise Exception("No files found after download")
        
        # Calculate downloaded size
        weight_files = list(local_path.rglob("*.safetensors")) + list(local_path.rglob("*.bin")) + list(local_path.rglob("*.pt")) + list(local_path.rglob("*.pth"))
        total_size = sum(f.stat().st_size for f in weight_files if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        download_status["models"][model_key].update({
            "status": "done",
            "progress": 100,
            "downloaded_mb": size_mb,
            "files_verified": True
        })
        
        print(f"✅ Successfully downloaded {model_id}")
        
    except Exception as e:
        download_status["models"][model_key]["status"] = "error"
        download_status["error"] = f"{model_key} failed: {str(e)}"
        print(f"❌ Failed to download {model_id}: {e}")
        raise e

def download_bark_models():
    """Download and setup Bark models"""
    global download_status
    
    try:
        print("🎵 Setting up Bark models...")
        download_status["models"]["bark"]["status"] = "downloading"
        download_status["current_model"] = "bark"
        download_status["message"] = "Setting up Bark models..."
        
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
                
                download_status["models"]["bark"].update({
                    "status": "done",
                    "progress": 100,
                    "downloaded_mb": size_mb,
                    "files_verified": True
                })
                print(f"✅ Bark models ready")
            else:
                raise Exception("Bark cache not found after preload")
                
        except ImportError:
            # Fallback: download Bark repository
            print("📦 Bark package not found, downloading repository...")
            safe_snapshot_download("suno/bark", "../models/audio/bark", "bark")
            
    except Exception as e:
        download_status["models"]["bark"]["status"] = "error"
        download_status["error"] = f"Bark setup failed: {str(e)}"
        print(f"❌ Bark setup failed: {e}")
        raise e

def download_all_models_background():
    """Background thread to download all models sequentially"""
    global download_status
    
    try:
        print("🚀 Starting unified model download...")
        download_status.update({
            "is_downloading": True,
            "status": "downloading",
            "overall_progress": 0,
            "error": None
        })
        
        # First, check existing models to get accurate status
        check_existing_models()
        
        # Reset all model statuses to pending for re-download
        for model_key in download_status["models"]:
            download_status["models"][model_key].update({
                "status": "pending",
                "progress": 0,
                "downloaded_mb": 0,
                "files_verified": False
            })
        
        # Download models sequentially
        models_to_download = []
        for model_key, model_info in download_status["models"].items():
            models_to_download.append((model_key, model_info))
        
        if not models_to_download:
            print("✅ All models already downloaded!")
            download_status.update({
                "is_downloading": False,
                "status": "completed",
                "overall_progress": 100,
                "message": "All models already downloaded!"
            })
            return
        
        total_models = len(models_to_download)
        
        for i, (model_key, model_info) in enumerate(models_to_download):
            try:
                if model_key == "bark":
                    download_bark_models()
                else:
                    safe_snapshot_download(
                        model_info["repo_id"],
                        model_info["local_dir"],
                        model_key
                    )
                
                # Update overall progress
                completed = i + 1
                download_status["overall_progress"] = int((completed / total_models) * 100)
                
            except Exception as e:
                print(f"❌ Failed to download {model_key}: {e}")
                download_status["models"][model_key]["status"] = "error"
                download_status["error"] = f"{model_key} download failed: {str(e)}"
                break
        
        # Check if all downloads completed successfully
        all_done = all(model["status"] == "done" for model in download_status["models"].values())
        
        if all_done:
            download_status.update({
                "is_downloading": False,
                "status": "completed",
                "overall_progress": 100,
                "message": "All models downloaded successfully!",
                "current_model": ""
            })
            
            # Generate voice previews after successful download
            try:
                print("🎤 Generating voice previews...")
                import asyncio
                asyncio.create_task(generate_voice_previews())
            except Exception as preview_error:
                print(f"⚠️  Voice preview generation failed: {preview_error}")
                
            print("🎉 All models downloaded successfully!")
        else:
            download_status.update({
                "is_downloading": False,
                "status": "error",
                "message": "Some models failed to download"
            })
            
    except Exception as e:
        download_status.update({
            "is_downloading": False,
            "status": "error",
            "message": f"Download failed: {str(e)}",
            "error": str(e)
        })
        print(f"❌ Download process failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global video_generator, audio_generator, image_generator
    
    # Detect GPU and set up models
    gpu_info = gpu_detector.detect_gpu()
    print(f"Detected GPU: {gpu_info}")
    
    # Initialize generators (but don't load models yet)
    video_generator = UltimateVideoGenerator(gpu_info)
    audio_generator = AudioGenerator(gpu_info)
    image_generator = ImageGenerator(gpu_info)
    
    print("Backend started successfully. Models will be loaded when requested.")
    print("Use the 'Download Models' button in the UI to download and load models.")
    
    # Check existing models on startup
    check_existing_models()

@app.get("/")
async def root():
    return {"message": "Text-to-Media Generator API", "version": "1.0.0"}

@app.get("/gpu-info")
async def get_gpu_info():
    """Get GPU information"""
    return gpu_detector.detect_gpu()

@app.get("/storage-usage")
async def get_storage_usage():
    """Return total bytes used by files in outputs directory"""
    outputs_dir = config.outputs_path
    total_bytes = 0
    try:
        if outputs_dir.exists():
            for path in outputs_dir.rglob('*'):
                if path.is_file():
                    try:
                        total_bytes += path.stat().st_size
                    except Exception:
                        continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"bytes": total_bytes}

@app.get("/settings")
async def get_settings():
    """Return system and app settings"""
    try:
        import psutil
        import platform
        import subprocess
        
        usage = await get_storage_usage()
        
        # Enhanced GPU detection for cross-platform
        gpu_info = gpu_detector.detect_gpu()
        
        # Get system information
        system_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }
        
        # Get memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 1),
            "available_gb": round(memory.available / (1024**3), 1),
            "percent": memory.percent,
        }
        
        # Check FFmpeg availability
        ffmpeg_available = False
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
            ffmpeg_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        return {
            "outputs_path": str(config.outputs_path),
            "storage_usage_bytes": usage["bytes"],
            "gpu_info": gpu_info,
            "ffmpeg_available": ffmpeg_available,
            "system_info": system_info,
            "memory_info": memory_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/outputs/clear")
async def clear_outputs():
    """Delete all files in outputs directory, except voice previews."""
    outputs_dir = config.outputs_path
    if not outputs_dir.exists():
        return {"message": "Outputs directory does not exist", "deleted_files": 0, "freed_space_mb": 0}
    
    try:
        deleted_files = 0
        total_size = 0
        
        # Delete all files in outputs directory, skipping voice previews
        for file_path in outputs_dir.rglob('*'):
            if file_path.is_file():
                # Skip files in voice previews directory
                try:
                    file_path.relative_to(config.voice_previews_path)
                    # Inside voice previews -> skip
                    continue
                except ValueError:
                    pass
                try:
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    file_path.unlink()
                    deleted_files += 1
                    print(f"Deleted output file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
                    continue
        
        # Also clear any subdirectories EXCEPT voice previews dir
        for subdir in outputs_dir.iterdir():
            if subdir.is_dir():
                # Preserve voice previews directory
                if subdir.resolve() == config.voice_previews_path.resolve():
                    continue
                try:
                    shutil.rmtree(subdir)
                    print(f"Deleted output directory: {subdir}")
                except Exception as e:
                    print(f"Error deleting directory {subdir}: {e}")
                    continue
        
        size_mb = total_size / (1024 * 1024)
        return {
            "message": f"Successfully deleted {deleted_files} output files",
            "deleted_files": deleted_files,
            "freed_space_mb": round(size_mb, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# File serving endpoints (must come before generic DELETE route)
@app.get("/outputs/videos/{filename}")
async def serve_video_file(filename: str, request: Request):
    """Serve video files with proper binary handling"""
    try:
        file_path = config.videos_output_path / filename
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Video file not found")
        # Handle HTTP Range requests for streaming
        # Determine media type based on extension
        lower_name = filename.lower()
        media_type = "video/mp4"
        if lower_name.endswith(".gif"):
            media_type = "image/gif"
        elif lower_name.endswith(".webm"):
            media_type = "video/webm"
        elif lower_name.endswith(".mov"):
            media_type = "video/quicktime"
        elif lower_name.endswith(".avi"):
            media_type = "video/x-msvideo"

        range_header = request.headers.get("range")
        if range_header:
            # Parse: bytes=start-end
            try:
                file_size = file_path.stat().st_size
                bytes_unit, byte_range = range_header.split("=")
                start_str, end_str = (byte_range or "0-").split("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                start = max(0, start)
                end = min(file_size - 1, end)

                chunk_size = end - start + 1
                def iter_file_range(path: pathlib.Path, start_pos: int, end_pos: int):
                    with open(path, "rb") as f:
                        f.seek(start_pos)
                        remaining = end_pos - start_pos + 1
                        while remaining > 0:
                            chunk = f.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            remaining -= len(chunk)
                            yield chunk

                headers = {
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(chunk_size),
                    "Content-Type": media_type,
                }
                return StreamingResponse(iter_file_range(file_path, start, end), status_code=206, headers=headers)
            except Exception:
                # Fall through to full response
                pass

        return FileResponse(path=str(file_path), media_type=media_type, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/audio/{filename}")
async def serve_audio_file(filename: str, request: Request):
    """Serve audio files with proper binary handling"""
    try:
        file_path = config.audio_output_path / filename
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        media_type = "audio/mpeg" if filename.lower().endswith('.mp3') else "audio/wav"

        # Handle HTTP Range requests for streaming
        range_header = request.headers.get("range")
        if range_header:
            try:
                file_size = file_path.stat().st_size
                bytes_unit, byte_range = range_header.split("=")
                start_str, end_str = (byte_range or "0-").split("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                start = max(0, start)
                end = min(file_size - 1, end)

                chunk_size = end - start + 1
                def iter_file_range(path: pathlib.Path, start_pos: int, end_pos: int):
                    with open(path, "rb") as f:
                        f.seek(start_pos)
                        remaining = end_pos - start_pos + 1
                        while remaining > 0:
                            chunk = f.read(min(1024 * 512, remaining))
                            if not chunk:
                                break
                            remaining -= len(chunk)
                            yield chunk

                headers = {
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(chunk_size),
                    "Content-Type": media_type,
                }
                return StreamingResponse(iter_file_range(file_path, start, end), status_code=206, headers=headers)
            except Exception:
                # Fall through to full response
                pass

        return FileResponse(path=str(file_path), media_type=media_type, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/voice-previews/{filename}")
async def serve_voice_preview(filename: str):
    """Serve voice preview files with proper binary handling"""
    try:
        file_path = config.voice_previews_path / filename
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Voice preview file not found")
        
        # Determine media type based on file extension
        media_type = "audio/wav"
        if filename.lower().endswith('.mp3'):
            media_type = "audio/mpeg"
        elif filename.lower().endswith('.wav'):
            media_type = "audio/wav"
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/audio/bark/preset-audios/{filename}")
async def serve_preset_audio(filename: str):
    """Serve Bark preset audio files with proper binary handling"""
    try:
        file_path = pathlib.Path(f'../models/audio/bark/preset-audios/{filename}')
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Preset audio file not found")
        
        # Determine media type based on file extension
        media_type = "audio/mpeg" if filename.endswith('.mp3') else "audio/wav"
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/outputs/{file_type}/{filename}")
async def delete_output_file(file_type: str, filename: str):
    """Delete a specific output file"""
    if file_type not in ['videos', 'audio', 'image']:
        raise HTTPException(status_code=400, detail="Invalid file type. Must be 'videos', 'audio', or 'image'")
    
    # Handle image files which are in the main outputs directory
    if file_type == 'image':
        file_path = config.image_output_path / filename
    elif file_type == 'videos':
        file_path = config.videos_output_path / filename
    elif file_type == 'audio':
        file_path = config.audio_output_path / filename
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    try:
        file_path.unlink()
        return {"message": f"File {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    global video_generator, audio_generator, image_generator
    
    # Initialize generators if they don't exist
    if not video_generator or not audio_generator or not image_generator:
        gpu_info = gpu_detector.detect_gpu()
        video_generator = UltimateVideoGenerator(gpu_info)
        audio_generator = AudioGenerator(gpu_info)
        image_generator = ImageGenerator(gpu_info)
    
    return {
        "video_models": video_generator.get_available_models() if video_generator else [],
        "image_models": image_generator.get_available_models() if image_generator else [],
        "audio_models": audio_generator.get_available_models() if audio_generator else []
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_media(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate video or audio from text prompt"""
    job_id = str(uuid.uuid4())
    
    try:
        # Validate request
        if request.model_type not in ["video", "audio", "image"]:
            raise HTTPException(status_code=400, detail="model_type must be 'video', 'audio' or 'image'")
        
        # Add job to queue
        job_data = {
            "prompt": request.prompt,
            "model_type": request.model_type,
            "model_name": request.model_name,
            "duration": request.duration,
            "sample_rate": request.sample_rate,
            "output_format": request.output_format,
            "created_at": datetime.now().isoformat()
        }
        
        # Add advanced video settings if provided
        if request.model_type == "video":
            if request.resolution is not None:
                job_data["resolution"] = request.resolution
            if request.num_frames is not None:
                job_data["num_frames"] = request.num_frames
            if request.num_inference_steps is not None:
                job_data["num_inference_steps"] = request.num_inference_steps
            if request.guidance_scale is not None:
                job_data["guidance_scale"] = request.guidance_scale
            if request.motion_scale is not None:
                job_data["motion_scale"] = request.motion_scale
            if request.fps is not None:
                job_data["fps"] = request.fps
            if request.seed is not None:
                job_data["seed"] = request.seed
        
        job_queue.add_job(job_id, job_data)
        
        # Start background generation
        if request.model_type == "video":
            # Always use SD + SVD pipeline for video generation
            background_tasks.add_task(generate_video, job_id, request)
        elif request.model_type == "audio":
            background_tasks.add_task(generate_audio, job_id, request)
        else:
            background_tasks.add_task(generate_image_task, job_id, request)
        
        return GenerationResponse(
            job_id=job_id,
            status="queued",
            message="Generation job started"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status and progress"""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        output_file=job.get("output_file"),
        error=job.get("error")
    )

@app.get("/jobs")
async def get_all_jobs():
    """Get all jobs"""
    return job_queue.get_all_jobs()

@app.get("/outputs/videos")
async def get_video_outputs():
    """Get list of generated video files"""
    try:
        videos_dir = config.videos_output_path
        if not videos_dir.exists():
            return {"videos": []}
        
        videos = []
        for file_path in videos_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.webm', '.gif']:
                videos.append({
                    "filename": file_path.name,
                    "path": f"/outputs/videos/{file_path.name}",
                    "size": file_path.stat().st_size,
                    "created": file_path.stat().st_ctime
                })
        
        # Sort by creation time (newest first)
        videos.sort(key=lambda x: x["created"], reverse=True)
        return {"videos": videos}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/audio")
async def get_audio_outputs():
    """Get list of generated audio files"""
    try:
        audio_dir = config.audio_output_path
        if not audio_dir.exists():
            return {"audio": []}
        
        audio_files = []
        for file_path in audio_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3']:
                audio_files.append({
                    "filename": file_path.name,
                    "path": f"/outputs/audio/{file_path.name}",
                    "size": file_path.stat().st_size,
                    "created": file_path.stat().st_ctime
                })
        
        # Sort by creation time (newest first)
        audio_files.sort(key=lambda x: x["created"], reverse=True)
        return {"audio": audio_files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/images")
async def get_image_outputs():
    """Get list of generated image files"""
    try:
        # Check both the main outputs directory and outputs/videos for images
        output_dirs = [
            config.image_output_path,
            config.videos_output_path
        ]
        
        image_files = []
        for output_dir in output_dirs:
            if not output_dir.exists():
                continue
                
            for file_path in output_dir.iterdir():
                # Treat GIFs as videos, not images
                if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    # Determine the correct path for serving
                    if output_dir.name == 'outputs':
                        path = f"/outputs/{file_path.name}"
                    else:
                        path = f"/outputs/videos/{file_path.name}"
                    
                    image_files.append({
                        "filename": file_path.name,
                        "path": path,
                        "size": file_path.stat().st_size,
                        "created": file_path.stat().st_ctime
                    })
        
        # Sort by creation time (newest first)
        image_files.sort(key=lambda x: x["created"], reverse=True)
        return {"images": image_files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/image/{filename}")
async def serve_image_file(filename: str):
    """Serve image files from the main outputs directory"""
    try:
        file_path = config.image_output_path / filename
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Determine media type based on file extension
        media_type = "image/png"  # default
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            media_type = "image/jpeg"
        elif filename.lower().endswith('.gif'):
            media_type = "image/gif"
        elif filename.lower().endswith('.webp'):
            media_type = "image/webp"
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voice-previews")
async def get_voice_previews():
    """Get list of available voice preview files from both generated and preset sources"""
    try:
        preview_files = []
        voice_preview_map = {}  # To avoid duplicates and prioritize mp3 over wav
        
        # First, check for Bark preset audio files (MP3 - preferred)
        preset_dir = pathlib.Path('../models/audio/bark/preset-audios')
        if preset_dir.exists():
            for file_path in preset_dir.glob("*-preview.mp3"):
                # Extract voice ID from filename like "v2_en_speaker_0-preview.mp3"
                voice_id = file_path.stem.replace("-preview", "").replace("_", "/")
                voice_preview_map[voice_id] = {
                    "voice_id": voice_id,
                    "voice_name": voice_id.replace('v2/', '').replace('_', ' ').title(),
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "url": f"/models/audio/bark/preset-audios/{file_path.name}",
                    "type": "preset"
                }
        
        # Then, check for generated voice previews (MP3 files only)
        previews_dir = pathlib.Path('../outputs/voice-previews')
        if previews_dir.exists():
            # Check for MP3 files (generated voice previews)
            for file_path in previews_dir.glob("*-preview.mp3"):
                voice_id = file_path.stem.replace('-preview', '').replace('_', '/')
                # Only add if we don't already have a preset MP3 version
                if voice_id not in voice_preview_map:
                    voice_preview_map[voice_id] = {
                        "voice_id": voice_id,
                        "voice_name": voice_id.replace('v2/', '').replace('_', ' ').title(),
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "url": f"/outputs/voice-previews/{file_path.name}",
                        "type": "generated"
                    }
        
        # Convert map to list and sort
        preview_files = list(voice_preview_map.values())
        return {"previews": sorted(preview_files, key=lambda x: x["voice_id"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bark-voices")
async def get_bark_voices():
    """Get list of available Bark voices using the English voice presets."""
    try:
        # Initialize audio generator if not already done
        if audio_generator is None:
            gpu_info = gpu_detector.get_gpu_info()
            audio_gen = AudioGenerator(gpu_info)
            voices = audio_gen.get_bark_voice_presets()
        else:
            voices = audio_generator.get_bark_voice_presets()
        
        return {"voices": voices}
    except Exception as e:
        # Fallback to hardcoded voices if there's an error
        voices = [
            {"id": "v2/en_speaker_0", "name": "Speaker 0 (EN)", "description": "English Male", "gender": "Male", "language": "English", "special": None},
            {"id": "v2/en_speaker_1", "name": "Speaker 1 (EN)", "description": "English Male", "gender": "Male", "language": "English", "special": None},
            {"id": "v2/en_speaker_2", "name": "Speaker 2 (EN)", "description": "English Male", "gender": "Male", "language": "English", "special": None},
            {"id": "v2/en_speaker_3", "name": "Speaker 3 (EN)", "description": "English Male", "gender": "Male", "language": "English", "special": None},
            {"id": "v2/en_speaker_4", "name": "Speaker 4 (EN)", "description": "English Male", "gender": "Male", "language": "English", "special": None},
            {"id": "v2/en_speaker_5", "name": "Speaker 5 (EN)", "description": "English Male", "gender": "Male", "language": "English", "special": "Grainy"},
            {"id": "v2/en_speaker_6", "name": "Speaker 6 (EN)", "description": "English Male", "gender": "Male", "language": "English", "special": "Suno Favorite"},
            {"id": "v2/en_speaker_7", "name": "Speaker 7 (EN)", "description": "English Male", "gender": "Male", "language": "English", "special": None},
            {"id": "v2/en_speaker_8", "name": "Speaker 8 (EN)", "description": "English Male", "gender": "Male", "language": "English", "special": None},
            {"id": "v2/en_speaker_9", "name": "Speaker 9 (EN)", "description": "English Female", "gender": "Female", "language": "English", "special": None}
        ]
        return {"voices": voices}

@app.get("/download-status")
async def get_download_status():
    """Get current download status"""
    return download_status

@app.post("/download-cleanup")
async def cleanup_download_status():
    """Reset download status (useful when download is interrupted)"""
    global download_status
    
    download_status.update({
        "is_downloading": False,
        "overall_progress": 0,
        "current_model": "",
        "status": "idle",
        "message": "",
        "error": None
    })
    
    # Reset model statuses
    for model_id in download_status["models"]:
        download_status["models"][model_id].update({
            "downloaded_mb": 0,
            "progress": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
            "status": "pending",
            "files_verified": False
        })
    
    return {"message": "Download status cleaned up successfully"}

@app.post("/optimize-models")
async def optimize_models():
    """Optimize models by removing duplicates and keeping only fp16 versions"""
    try:
        import subprocess
        import sys
        
        # Run the optimization script
        result = subprocess.run([
            sys.executable, 
            "scripts/optimize-models.py"
        ], capture_output=True, text=True, cwd="..")
        
        if result.returncode == 0:
            return {
                "success": True,
                "message": "Models optimized successfully",
                "output": result.stdout
            }
        else:
            return {
                "success": False,
                "message": "Optimization failed",
                "error": result.stderr
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Optimization error: {str(e)}"
        }

@app.post("/download-models")
async def download_models(background_tasks: BackgroundTasks, force: bool = False):
    """Start downloading all models using unified downloader"""
    global download_status
    
    if download_status["is_downloading"]:
        raise HTTPException(status_code=400, detail="Download already in progress")
    
    # If force is True, reset all model statuses to pending
    if force:
        print("🔄 Force re-download requested - resetting all model statuses")
        for model_key in download_status["models"]:
            download_status["models"][model_key].update({
                "status": "pending",
                "progress": 0,
                "downloaded_mb": 0,
                "files_verified": False
            })
    
    # Start download in background thread
    download_thread = threading.Thread(target=download_all_models_background, daemon=True)
    download_thread.start()
    
    return {"message": "Unified model download started", "status": "downloading", "force": force}


@app.post("/load-models")
async def load_models():
    """Load models after download is complete"""
    try:
        global video_generator, audio_generator
        
        print("Loading models after download completion...")
        
        # Load video models
        await video_generator.load_default_models()
        
        # Load audio models  
        await audio_generator.load_default_models()
        
        return {"message": "Models loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@app.get("/download-models-stream")
async def download_models_stream(force: bool = False, retries: int = 5):
    """Stream download progress in real-time"""
    
    def generate_download_logs():
        """Generator function that yields download logs in real-time"""
        try:
            # Prepare the download command
            script_path = pathlib.Path(__file__).parent.parent / "scripts" / "download-models.py"
            cmd = [sys.executable, str(script_path), "--all", "--retries", str(retries)]
            
            if force:
                cmd.append("--force")
            
            # Start the download process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Mirror to backend terminal for visibility
                    try:
                        print(line.strip())
                    except Exception:
                        pass
                    # Format as Server-Sent Events
                    yield f"data: {json.dumps({'type': 'log', 'message': line.strip()})}\n\n"
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                yield f"data: {json.dumps({'type': 'success', 'message': 'Download completed successfully!'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Download failed with exit code {process.returncode}'})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Download error: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        generate_download_logs(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

# REMOVED: Individual model download endpoint
# @app.get("/download-model-stream/{model_name}")
# async def download_model_stream(model_name: str, force: bool = False, retries: int = 5):
#     """Stream download progress for a specific model"""
#     # ... entire function removed ...

@app.delete("/delete-model/{model_name}")
async def delete_model(model_name: str):
    """Delete a specific model"""
    import shutil
    from fastapi import HTTPException as _HTTPException
    try:
        # Define model paths
        model_paths = {
            "stable-diffusion": pathlib.Path("../models/image/stable-diffusion"),
            "svd": pathlib.Path("../models/video/svd"),
            "realesrgan": pathlib.Path("../models/upscaling/realesrgan"),
            "enhanced-pipeline": None,  # Special case - will handle grouped deletion
            # Bark stores in cache; also keep a local mirror under ../models/audio/bark
            "bark": pathlib.Path.home() / ".cache" / "suno" / "bark_v0"
        }
        # Map API names to unified download status keys
        name_to_status_key = {
            "stable-diffusion": "stable_diffusion",
            "svd": "svd",
            "realesrgan": "realesrgan",
            "enhanced-pipeline": "enhanced_pipeline",
            "bark": "bark",
        }

        if model_name not in model_paths:
            # Return a proper 400 for unknown/undefined names
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

        # Handle enhanced-pipeline deletion (delete all grouped models)
        if model_name == "enhanced-pipeline":
            deleted_models = []
            enhanced_pipeline_models = ["svd", "realesrgan"]
            total_size_gb = 0
            
            for sub_model in enhanced_pipeline_models:
                sub_path = model_paths[sub_model]
                if sub_path.exists():
                    # Calculate size before deletion
                    sub_size = 0
                    if sub_path.is_dir():
                        for file_path in sub_path.rglob('*'):
                            if file_path.is_file():
                                sub_size += file_path.stat().st_size
                    total_size_gb += sub_size / (1024**3)
                    
                    # Delete the model
                    shutil.rmtree(sub_path)
                    deleted_models.append(sub_model)
                    print(f"Deleted model: {sub_model}")
                    
                    # Update download status for each sub-model
                    status_key = name_to_status_key.get(sub_model)
                    if status_key:
                        download_status[status_key] = "not_downloaded"
            
            if deleted_models:
                return {
                    "message": f"Enhanced pipeline deleted successfully. Removed: {', '.join(deleted_models)}",
                    "deleted": True,
                    "size_gb": round(total_size_gb, 2)
                }
            else:
                return {"message": "Enhanced pipeline models not found", "deleted": False}

        model_path = model_paths[model_name]

        # Special handling for Bark: delete both cache and local mirror
        deleted_any = False
        if model_name == "bark":
            bark_cache = model_path
            bark_local = pathlib.Path("../models/audio/bark")
            # Delete cache dir
            if bark_cache.exists():
                if bark_cache.is_dir():
                    shutil.rmtree(bark_cache)
                else:
                    bark_cache.unlink()
                deleted_any = True
            # Delete local mirror dir
            if bark_local.exists():
                shutil.rmtree(bark_local)
                deleted_any = True
            if not deleted_any:
                return {"message": "Bark not found", "deleted": False}
            # For size freed, we cannot easily compute both after deletion; report 0 and rely on /settings storage
            size_gb = 0
        else:
            if not model_path.exists():
                return {"message": f"Model {model_name} not found", "deleted": False}
            
            # Calculate size before deletion
            total_size = 0
            if model_path.is_dir():
                for file_path in model_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

            size_gb = total_size / (1024**3)

            # Delete the model
            if model_path.is_dir():
                shutil.rmtree(model_path)
            else:
                model_path.unlink()

        # Update unified download status for the deleted model
        try:
            status_key = name_to_status_key.get(model_name, model_name)
            if status_key in download_status.get("models", {}):
                download_status["models"][status_key].update({
                    "status": "pending",
                    "progress": 0,
                    "downloaded_mb": 0,
                    "files_verified": False
                })
                # Recompute overall progress
                completed_models = sum(1 for m in download_status["models"].values() if m.get("status") == "done")
                total_models = len(download_status["models"]) or 1
                download_status["overall_progress"] = int((completed_models / total_models) * 100)
        except Exception:
            pass

        return {
            "message": f"Model {model_name} deleted successfully",
            "deleted": True,
            "size_freed_gb": round(size_gb, 2)
        }
    except _HTTPException as http_err:
        # Preserve intended HTTP status codes (e.g., 400 Unknown model)
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model {model_name}: {str(e)}")

@app.post("/cancel-download/{model_name}")
async def cancel_download(model_name: str):
    """Cancel an in-progress model download and delete partial files."""
    try:
        info = download_status["models"].get(model_name)
        if not info:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

        # Best effort: delete local dir contents
        local_dir = pathlib.Path(info["local_dir"]) if model_name != 'bark' else (pathlib.Path.home() / '.cache' / 'suno' / 'bark_v0')
        if local_dir.exists():
            try:
                import shutil as _shutil
                _shutil.rmtree(local_dir)
            except Exception:
                pass

        # Reset status
        info.update({
            'status': 'pending',
            'progress': 0,
            'downloaded_mb': 0,
            'files_verified': False
        })
        download_status.update({
            'current_model': '',
            'message': '',
            'is_downloading': False
        })
        return { 'message': f'Cancelled {model_name} download and cleaned partial files.' }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel download: {e}")

def restart_models():
    """Restart model loading after download"""
    global video_generator, audio_generator
    
    try:
        # Reinitialize generators
        gpu_info = gpu_detector.detect_gpu()
        video_generator = UltimateVideoGenerator(gpu_info)
        audio_generator = AudioGenerator(gpu_info)
        
        # Load models
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(video_generator.load_default_models())
        loop.run_until_complete(audio_generator.load_default_models())
        loop.close()
        
    except Exception as e:
        print(f"Error restarting models: {e}")

async def generate_voice_previews():
    """Generate voice preview samples for each Bark voice"""
    try:
        print("🎤 Generating voice preview samples...")
        
        # Create voice previews directory
        previews_dir = pathlib.Path('../outputs/voice-previews')
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
            from pydub import AudioSegment
            import numpy as np
            
            print("✅ Bark imported successfully, generating voice previews...")
            
            for voice_id, sample_text in voice_samples.items():
                try:
                    print(f"🎵 Generating preview for {voice_id}...")
                    
                    # Check if preview already exists
                    preview_filename = f"{voice_id.replace('/', '_')}-preview.mp3"
                    preview_path = previews_dir / preview_filename
                    
                    if preview_path.exists():
                        print(f"⏭️  Preview for {voice_id} already exists, skipping...")
                        continue
                    
                    # Generate audio with specific voice
                    audio_array = generate_audio(sample_text, history_prompt=voice_id)
                    
                    # Save preview file as MP3 for better browser compatibility
                    if isinstance(audio_array, np.ndarray):
                        # First save as temporary WAV
                        temp_wav_path = previews_dir / f"temp_{voice_id.replace('/', '_')}.wav"
                        sf.write(str(temp_wav_path), audio_array, SAMPLE_RATE)
                        
                        # Convert to MP3 using pydub
                        audio_segment = AudioSegment.from_wav(str(temp_wav_path))
                        audio_segment.export(str(preview_path), format="mp3", bitrate="128k")
                        
                        # Remove the temporary WAV file
                        temp_wav_path.unlink()
                        
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

@app.post("/save-custom-voice")
async def save_custom_voice(voice_name: str, voice_data: bytes = File(...)):
    """Save a custom XTTS voice recording"""
    try:
        # Create custom voices directory
        custom_voices_dir = pathlib.Path("../outputs/custom-voices")
        custom_voices_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the voice file
        voice_filename = f"{voice_name.replace(' ', '_').lower()}.wav"
        voice_path = custom_voices_dir / voice_filename
        
        with open(voice_path, "wb") as f:
            f.write(voice_data)
        
        return {"message": f"Custom voice '{voice_name}' saved successfully", "voice_path": str(voice_path)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save custom voice: {str(e)}")

@app.get("/custom-voices")
async def get_custom_voices():
    """Get list of saved custom voices"""
    try:
        custom_voices_dir = pathlib.Path("../outputs/custom-voices")
        if not custom_voices_dir.exists():
            return {"voices": []}
        
        voices = []
        for voice_file in custom_voices_dir.glob("*.wav"):
            voice_name = voice_file.stem.replace('_', ' ').title()
            voices.append({
                "id": voice_file.stem,
                "name": voice_name,
                "filename": voice_file.name,
                "path": str(voice_file),
                "size": voice_file.stat().st_size
            })
        
        return {"voices": voices}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get custom voices: {str(e)}")

@app.get("/custom-voices/{filename}")
async def serve_custom_voice(filename: str):
    """Serve a custom voice file"""
    try:
        custom_voices_dir = pathlib.Path("../outputs/custom-voices")
        voice_file = custom_voices_dir / filename
        
        if not voice_file.exists() or not voice_file.is_file():
            raise HTTPException(status_code=404, detail="Voice file not found")
        
        return FileResponse(
            path=str(voice_file),
            media_type="audio/wav",
            filename=filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve voice file: {str(e)}")


@app.post("/generate-voice-previews")
async def generate_voice_previews_endpoint():
    """Generate voice preview samples"""
    try:
        await generate_voice_previews()
        return {"message": "Voice previews generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate voice previews: {str(e)}")

@app.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job"""
    success = job_queue.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job cancelled"}

@app.delete("/jobs")
async def clear_all_jobs():
    """Clear all jobs from the queue"""
    count = job_queue.clear_all_jobs()
    return {"message": f"Cleared {count} jobs", "cleared_count": count}

@app.post("/models/{model_type}/{model_name}/load")
async def load_model(model_type: str, model_name: str):
    """Load a specific model"""
    try:
        if model_type not in ["video", "audio"]:
            raise HTTPException(status_code=400, detail="model_type must be 'video' or 'audio'")
        
        if model_type == "video":
            success = await video_generator.load_model(model_name)
        else:
            success = await audio_generator.load_model(model_name)
        
        if success:
            return {"message": f"Model {model_name} loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-voice")
async def upload_voice(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """Upload a custom reference voice WAV/MP3 to be used for XTTS cloning.
    Saves under ../models/audio/bark/preset-audios and returns the absolute path.
    """
    try:
        import re
        base_dir = pathlib.Path("../models/audio/bark/preset-audios")
        base_dir.mkdir(parents=True, exist_ok=True)

        original = pathlib.Path(file.filename or "voice.wav").name
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", (name or pathlib.Path(original).stem))
        ext = pathlib.Path(original).suffix.lower() or ".wav"
        # Force wav extension for better compatibility
        if ext not in [".wav", ".mp3", ".m4a", ".flac"]:
            ext = ".wav"
        out_path = base_dir / f"{safe_name}{ext}"

        contents = await file.read()
        with open(out_path, "wb") as f:
            f.write(contents)

        return {
            "success": True,
            "path": str(out_path.resolve()),
            "filename": out_path.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload voice: {str(e)}")

@app.post("/models/{model_type}/{model_name}/unload")
async def unload_model(model_type: str, model_name: str):
    """Unload a specific model"""
    try:
        if model_type not in ["video", "audio"]:
            raise HTTPException(status_code=400, detail="model_type must be 'video' or 'audio'")
        
        if model_type == "video":
            success = video_generator.unload_model(model_name)
        else:
            success = audio_generator.unload_model(model_name)
        
        if success:
            return {"message": f"Model {model_name} unloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to unload model {model_name}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-models")
async def delete_models():
    """Delete all downloaded models"""
    try:
        import shutil
        
        # Define model directories
        models_dir = pathlib.Path('../models')
        
        if not models_dir.exists():
            return {"message": "No models directory found"}
        
        # List of model directories to delete
        model_dirs = [
            ('image', 'stable-diffusion'), 
            ('video', 'svd'),
            ('upscaling', 'realesrgan'),
            ('audio', 'bark'),
            ('', 'huggingface')  # Common cache directory
        ]
        
        deleted_count = 0
        total_size = 0
        
        for subdir, model_dir in model_dirs:
            if subdir:
                model_path = models_dir / subdir / model_dir
            else:
                model_path = models_dir / model_dir
                
            if model_path.exists() and model_path.is_dir():
                try:
                    # Calculate size before deletion
                    for file_path in model_path.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                    
                    # Delete the directory
                    shutil.rmtree(model_path)
                    deleted_count += 1
                    print(f"Deleted model directory: {model_path}")
                except Exception as e:
                    print(f"Error deleting {model_path}: {e}")
                    continue
        
        # Also clear any cache directories
        cache_dirs = [
            pathlib.Path.home() / '.cache' / 'huggingface',
            pathlib.Path.home() / '.cache' / 'torch',
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    for file_path in cache_dir.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                    shutil.rmtree(cache_dir)
                    print(f"Deleted cache directory: {cache_dir}")
                except Exception as e:
                    print(f"Error deleting cache {cache_dir}: {e}")
        
        # Clear model registry after deletion
        global video_generator, audio_generator
        video_generator = None
        audio_generator = None
        
        size_mb = total_size / (1024 * 1024)
        return {
            "message": f"Successfully deleted {deleted_count} model directories",
            "deleted_directories": deleted_count,
            "freed_space_mb": round(size_mb, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete models: {str(e)}")

@app.post("/clear-outputs")
async def clear_outputs():
    """Clear all output files (videos and audios)"""
    try:
        import shutil
        import os
        
        # Get outputs directory from settings
        outputs_dir = pathlib.Path('../outputs')
        if not outputs_dir.exists():
            return {"message": "No outputs directory found", "deleted_files": 0, "freed_space_mb": 0}
        
        deleted_files = 0
        total_size = 0
        
        # Delete all files in outputs directory
        for file_path in outputs_dir.rglob('*'):
            if file_path.is_file():
                try:
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    file_path.unlink()
                    deleted_files += 1
                    print(f"Deleted output file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
                    continue
        
        # Also clear any subdirectories
        for subdir in outputs_dir.iterdir():
            if subdir.is_dir():
                try:
                    shutil.rmtree(subdir)
                    print(f"Deleted output directory: {subdir}")
                except Exception as e:
                    print(f"Error deleting directory {subdir}: {e}")
                    continue
        
        size_mb = total_size / (1024 * 1024)
        return {
            "message": f"Successfully deleted {deleted_files} output files",
            "deleted_files": deleted_files,
            "freed_space_mb": round(size_mb, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear outputs: {str(e)}")

async def generate_video(job_id: str, request: GenerationRequest):
    """Generate video in background"""
    try:
        job_queue.update_job(job_id, {"status": "processing", "progress": 5})
        
        # Update progress: Loading model
        job_queue.update_job(job_id, {"progress": 15, "message": "Loading video model..."})
        
        # Generate video with progress callbacks
        def progress_callback(progress: int, message: str = ""):
            job_queue.update_job(job_id, {"progress": progress, "message": message})
        
        # Prepare generation parameters - always use SD + SVD pipeline
        generation_params = {
            "prompt": request.prompt,
            "preset": "balanced",  # Use balanced preset for SD + SVD pipeline
            "output_format": request.output_format,
            "progress_callback": progress_callback
        }
        
        # Add advanced settings if provided
        if request.resolution is not None:
            generation_params["resolution"] = request.resolution
        if request.seed is not None:
            generation_params["seed"] = request.seed
        
        # Create SVD overrides for advanced settings
        svd_overrides = {}
        if request.num_frames is not None:
            svd_overrides["frames"] = request.num_frames
        if request.num_inference_steps is not None:
            svd_overrides["steps"] = request.num_inference_steps
        if request.motion_scale is not None:
            svd_overrides["motion_bucket_id"] = int(127 * request.motion_scale)
        if request.fps is not None:
            svd_overrides["fps"] = request.fps
        
        if svd_overrides:
            generation_params["svd_overrides"] = svd_overrides
        
        output_path = await video_generator.generate(**generation_params)
        
        job_queue.update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "output_file": output_path,
            "message": "Video generation completed!"
        })
        
    except Exception as e:
        job_queue.update_job(job_id, {
            "status": "failed",
            "error": str(e),
            "message": f"Video generation failed: {str(e)}"
        })

async def generate_audio(job_id: str, request: GenerationRequest):
    """Generate audio in background"""
    try:
        job_queue.update_job(job_id, {"status": "processing", "progress": 10})
        
        # Generate audio
        output_path = await audio_generator.generate(
            prompt=request.prompt,
            model_name=request.model_name,
            sample_rate=request.sample_rate,
            output_format=request.output_format,
            voice_style=request.voice_style,
            voice_id=request.voice_id
        )
        
        job_queue.update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "output_file": output_path
        })
        
    except Exception as e:
        job_queue.update_job(job_id, {
            "status": "failed",
            "error": str(e)
        })

async def generate_image_task(job_id: str, request: GenerationRequest):
    """Generate image in background"""
    try:
        job_queue.update_job(job_id, {"status": "processing", "progress": 10})
        
        def progress_callback(progress: int, message: str = ""):
            job_queue.update_job(job_id, {"progress": progress, "message": message})
        
        # Use dedicated image generator
        global image_generator
        if not image_generator:
            gpu_info = gpu_detector.detect_gpu()
            image_generator = ImageGenerator(gpu_info)
        
        # Extract image generation parameters
        resolution = request.resolution if request.resolution is not None else 512
        num_inference_steps = request.num_inference_steps if request.num_inference_steps is not None else 50
        guidance_scale = request.guidance_scale if request.guidance_scale is not None else 9.0
        
        output_path = await image_generator.generate_image(
            prompt=request.prompt,
            model_name=request.model_name,
            width=resolution,
            height=resolution,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            progress_callback=progress_callback
        )
        
        job_queue.update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "output_file": output_path,
            "message": "Image generation completed!"
        })
    except Exception as e:
        job_queue.update_job(job_id, {
            "status": "failed",
            "error": str(e),
            "message": f"Image generation failed: {str(e)}"
        })

# Mount static files for serving generated media (after all API routes)
# Removed static file mount to avoid Unicode encoding issues with binary files
# Using FileResponse endpoints instead for proper binary file handling

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
