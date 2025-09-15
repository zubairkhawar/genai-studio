from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import json
from typing import List, Optional
from pydantic import BaseModel
import asyncio
from datetime import datetime
import time
import uuid
import pathlib

from models.video_generator import VideoGenerator
from models.audio_generator import AudioGenerator
from utils.gpu_detector import GPUDetector
from utils.ffmpeg_handler import FFmpegHandler
import shutil
from utils.job_queue import JobQueue
import subprocess
import threading
import time
import os
from huggingface_hub import snapshot_download

app = FastAPI(title="Text-to-Media Generator", version="1.0.0")

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

# Initialize components
gpu_detector = GPUDetector()
ffmpeg_handler = FFmpegHandler()
job_queue = JobQueue()

# Global model instances
video_generator = None
audio_generator = None

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
            "size_gb": 4.0,
            "status": "pending",  # pending, downloading, done, error
            "progress": 0,
            "downloaded_mb": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
            "files_verified": False
        },
        "stable_video_diffusion": {
            "name": "Stable Video Diffusion",
            "repo_id": "stabilityai/stable-video-diffusion-img2vid", 
            "local_dir": "../models/video/stable-video-diffusion",
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

class GenerationRequest(BaseModel):
    prompt: str
    model_type: str  # "video" or "audio"
    model_name: str
    duration: Optional[int] = 5  # for video
    sample_rate: Optional[int] = 22050  # for audio
    output_format: str = "mp4"  # mp4, wav, mp3
    voice_style: Optional[str] = "auto"  # for audio voice selection
    voice_id: Optional[str] = None  # specific Bark voice ID

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
        
        if local_dir.exists():
            # Check for actual model weight files
            weight_files = list(local_dir.rglob("*.safetensors")) + list(local_dir.rglob("*.bin")) + list(local_dir.rglob("*.pt")) + list(local_dir.rglob("*.pth"))
            
            if len(weight_files) > 0:
                # Calculate total size
                total_size = sum(f.stat().st_size for f in weight_files if f.is_file())
                size_mb = total_size / (1024 * 1024)
                
                download_status["models"][model_key].update({
                    "status": "done",
                    "progress": 100,
                    "downloaded_mb": size_mb,
                    "files_verified": True
                })
                print(f"✅ {model_info['name']} already downloaded ({size_mb:.1f} MB)")
            else:
                print(f"⚠️  {model_info['name']} directory exists but no model files found")
        else:
            print(f"❌ {model_info['name']} not found")
    
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
        
        print(f"✅ Successfully downloaded {model_id} ({size_mb:.1f} MB)")
        
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
                print(f"✅ Bark models ready ({size_mb:.1f} MB)")
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
        
        # Reset all model statuses
        for model_key in download_status["models"]:
            if download_status["models"][model_key]["status"] != "done":
                download_status["models"][model_key].update({
                    "status": "pending",
                    "progress": 0,
                    "downloaded_mb": 0,
                    "files_verified": False
                })
        
        # Download models sequentially
        models_to_download = []
        for model_key, model_info in download_status["models"].items():
            if model_info["status"] != "done":
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
    global video_generator, audio_generator
    
    # Detect GPU and set up models
    gpu_info = gpu_detector.detect_gpu()
    print(f"Detected GPU: {gpu_info}")
    
    # Initialize generators (but don't load models yet)
    video_generator = VideoGenerator(gpu_info)
    audio_generator = AudioGenerator(gpu_info)
    
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
    outputs_dir = pathlib.Path('../outputs')
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
            "outputs_path": str(pathlib.Path('../outputs').resolve()),
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
    """Delete all files in outputs directory"""
    outputs_dir = pathlib.Path('../outputs')
    if not outputs_dir.exists():
        return {"message": "Outputs directory does not exist", "deleted_files": 0, "freed_space_mb": 0}
    
    try:
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
        raise HTTPException(status_code=500, detail=str(e))

# File serving endpoints (must come before generic DELETE route)
@app.get("/outputs/videos/{filename}")
async def serve_video_file(filename: str):
    """Serve video files with proper binary handling"""
    try:
        file_path = pathlib.Path(f'../outputs/videos/{filename}')
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        return FileResponse(
            path=str(file_path),
            media_type="video/mp4",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/audio/{filename}")
async def serve_audio_file(filename: str):
    """Serve audio files with proper binary handling"""
    try:
        file_path = pathlib.Path(f'../outputs/audio/{filename}')
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            path=str(file_path),
            media_type="audio/wav",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/voice-previews/{filename}")
async def serve_voice_preview(filename: str):
    """Serve voice preview files with proper binary handling"""
    try:
        file_path = pathlib.Path(f'../outputs/voice-previews/{filename}')
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Voice preview file not found")
        
        return FileResponse(
            path=str(file_path),
            media_type="audio/wav",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/outputs/{file_type}/{filename}")
async def delete_output_file(file_type: str, filename: str):
    """Delete a specific output file"""
    if file_type not in ['videos', 'audio']:
        raise HTTPException(status_code=400, detail="Invalid file type. Must be 'videos' or 'audio'")
    
    file_path = pathlib.Path(f'../outputs/{file_type}/{filename}')
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
    global video_generator, audio_generator
    
    # Initialize generators if they don't exist
    if not video_generator or not audio_generator:
        gpu_info = gpu_detector.detect_gpu()
        video_generator = VideoGenerator(gpu_info)
        audio_generator = AudioGenerator(gpu_info)
    
    return {
        "video_models": video_generator.get_available_models() if video_generator else [],
        "audio_models": audio_generator.get_available_models() if audio_generator else []
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_media(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate video or audio from text prompt"""
    job_id = str(uuid.uuid4())
    
    try:
        # Validate request
        if request.model_type not in ["video", "audio"]:
            raise HTTPException(status_code=400, detail="model_type must be 'video' or 'audio'")
        
        # Add job to queue
        job_queue.add_job(job_id, {
            "prompt": request.prompt,
            "model_type": request.model_type,
            "model_name": request.model_name,
            "duration": request.duration,
            "sample_rate": request.sample_rate,
            "output_format": request.output_format,
            "created_at": datetime.now().isoformat()
        })
        
        # Start background generation
        if request.model_type == "video":
            background_tasks.add_task(generate_video, job_id, request)
        else:
            background_tasks.add_task(generate_audio, job_id, request)
        
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
        videos_dir = pathlib.Path('../outputs/videos')
        if not videos_dir.exists():
            return {"videos": []}
        
        videos = []
        for file_path in videos_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.mp4', '.avi', '.mov', '.webm']:
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
        audio_dir = pathlib.Path('../outputs/audio')
        if not audio_dir.exists():
            return {"audio": []}
        
        audio_files = []
        for file_path in audio_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.wav', '.mp3', '.m4a', '.flac']:
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

@app.get("/voice-previews")
async def get_voice_previews():
    """Get list of available voice preview files"""
    try:
        previews_dir = pathlib.Path('../outputs/voice-previews')
        if not previews_dir.exists():
            return {"previews": []}
        
        preview_files = []
        for file_path in previews_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.wav':
                # Extract voice ID from filename (e.g., "v2_en_speaker_0-preview.wav" -> "v2/en_speaker_0")
                voice_id = file_path.stem.replace('-preview', '').replace('_', '/')
                preview_files.append({
                    "voice_id": voice_id,
                    "voice_name": voice_id.replace('v2/', '').replace('_', ' ').title(),
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "url": f"/outputs/voice-previews/{file_path.name}"
                })
        
        return {"previews": sorted(preview_files, key=lambda x: x["voice_id"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bark-voices")
async def get_bark_voices():
    """Get list of available Bark voices"""
    try:
        # Define available Bark voices with descriptions
        bark_voices = [
            {
                "id": "v2/en_speaker_0",
                "name": "Default Speaker",
                "description": "Clear and natural default English voice",
                "type": "default"
            },
            {
                "id": "v2/en_speaker_1", 
                "name": "Alternative Speaker",
                "description": "Alternative English speaker with distinct tone",
                "type": "alternative"
            },
            {
                "id": "v2/en_speaker_2",
                "name": "Warm Speaker", 
                "description": "Warm and friendly character voice",
                "type": "friendly"
            },
            {
                "id": "v2/en_speaker_3",
                "name": "Professional Speaker",
                "description": "Professional and authoritative tone",
                "type": "professional"
            },
            {
                "id": "v2/en_speaker_4",
                "name": "Storyteller",
                "description": "Perfect for storytelling and narration",
                "type": "narrator"
            },
            {
                "id": "v2/en_speaker_5",
                "name": "Casual Speaker",
                "description": "Casual and conversational style",
                "type": "casual"
            },
            {
                "id": "v2/en_speaker_6",
                "name": "Female Voice",
                "description": "Clear and engaging female voice",
                "type": "female"
            },
            {
                "id": "v2/en_speaker_7",
                "name": "Male Voice", 
                "description": "Deep and characterful male voice",
                "type": "male"
            },
            {
                "id": "v2/en_speaker_8",
                "name": "Young Voice",
                "description": "Energetic voice perfect for children's content",
                "type": "child"
            },
            {
                "id": "v2/en_speaker_9",
                "name": "Narrator Voice",
                "description": "Clear and authoritative narrator voice",
                "type": "narrator"
            }
        ]
        
        return {"voices": bark_voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/download-models")
async def download_models(background_tasks: BackgroundTasks):
    """Start downloading all models using unified downloader"""
    global download_status
    
    if download_status["is_downloading"]:
        raise HTTPException(status_code=400, detail="Download already in progress")
    
    # Start download in background thread
    download_thread = threading.Thread(target=download_all_models_background, daemon=True)
    download_thread.start()
    
    return {"message": "Unified model download started", "status": "downloading"}


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

def restart_models():
    """Restart model loading after download"""
    global video_generator, audio_generator
    
    try:
        # Reinitialize generators
        gpu_info = gpu_detector.detect_gpu()
        video_generator = VideoGenerator(gpu_info)
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

@app.get("/bark-voices")
async def get_bark_voices():
    """Get list of available Bark voices"""
    voices = [
        {
            "id": "v2/en_speaker_0",
            "name": "Male Speaker 1",
            "description": "Clear male voice with neutral tone",
            "type": "male"
        },
        {
            "id": "v2/en_speaker_1", 
            "name": "Female Speaker 1",
            "description": "Warm female voice with friendly tone",
            "type": "female"
        },
        {
            "id": "v2/en_speaker_2",
            "name": "Male Speaker 2", 
            "description": "Deep male voice with authoritative tone",
            "type": "male"
        },
        {
            "id": "v2/en_speaker_3",
            "name": "Female Speaker 2",
            "description": "Bright female voice with energetic tone", 
            "type": "female"
        },
        {
            "id": "v2/en_speaker_4",
            "name": "Male Speaker 3",
            "description": "Smooth male voice with calm tone",
            "type": "male"
        },
        {
            "id": "v2/en_speaker_5",
            "name": "Female Speaker 3", 
            "description": "Professional female voice with clear articulation",
            "type": "female"
        },
        {
            "id": "v2/en_speaker_6",
            "name": "Male Speaker 4",
            "description": "Casual male voice with relaxed tone",
            "type": "male"
        },
        {
            "id": "v2/en_speaker_7",
            "name": "Female Speaker 4",
            "description": "Gentle female voice with soothing tone",
            "type": "female"
        },
        {
            "id": "v2/en_speaker_8",
            "name": "Male Speaker 5",
            "description": "Confident male voice with strong presence",
            "type": "male"
        },
        {
            "id": "v2/en_speaker_9",
            "name": "Female Speaker 5",
            "description": "Expressive female voice with dynamic range",
            "type": "female"
        }
    ]
    
    return {"voices": voices}

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
            ('video', 'stable-video-diffusion'),
            ('image', 'stable-diffusion'), 
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
        
        output_path = await video_generator.generate(
            prompt=request.prompt,
            model_name=request.model_name,
            duration=request.duration,
            output_format=request.output_format,
            progress_callback=progress_callback
        )
        
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

# Mount static files for serving generated media (after all API routes)
# Removed static file mount to avoid Unicode encoding issues with binary files
# Using FileResponse endpoints instead for proper binary file handling

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
