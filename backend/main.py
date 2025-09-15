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

# Download status tracking
download_status = {
    "is_downloading": False,
    "progress": 0,
    "current_model": "",
    "status": "idle",  # idle, downloading, completed, error
    "message": "",
    "error": None,
    "models": {
        "stable-video-diffusion": {
            "name": "Stable Video Diffusion",
            "size_gb": 5.0,
            "downloaded_mb": 0,
            "progress": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
            "status": "pending"
        },
        "stable-diffusion": {
            "name": "Stable Diffusion", 
            "size_gb": 4.0,
            "downloaded_mb": 0,
            "progress": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
            "status": "pending"
        },
        "bark": {
            "name": "Bark",
            "size_gb": 5.0,
            "downloaded_mb": 0,
            "progress": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
            "status": "pending"
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
        "progress": 0,
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
            "status": "pending"
        })
    
    return {"message": "Download status cleaned up successfully"}

@app.post("/download-models")
async def download_models(background_tasks: BackgroundTasks):
    """Start downloading all models"""
    global download_status
    
    if download_status["is_downloading"]:
        raise HTTPException(status_code=400, detail="Download already in progress")
    
    # Reset status
    download_status.update({
        "is_downloading": True,
        "progress": 0,
        "current_model": "",
        "status": "downloading",
        "message": "Starting model download...",
        "error": None
    })
    
    # Reset model statuses
    for model_id in download_status["models"]:
        download_status["models"][model_id].update({
            "downloaded_mb": 0,
            "progress": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
            "status": "pending"
        })
    
    # Start download in background
    background_tasks.add_task(download_models_background)
    
    return {"message": "Model download started", "status": "downloading"}

def download_models_background():
    """Background task to download models"""
    global download_status
    
    try:
        # Get the Python script path (prefer Python script over shell script)
        script_path = pathlib.Path(__file__).parent.parent / "scripts" / "download-models.py"
        
        if not script_path.exists():
            raise Exception("Download script not found")
        
        # Models to download
        models = [
            {"name": "stable-video-diffusion", "type": "video"},
            {"name": "stable-diffusion", "type": "image"}, 
            {"name": "bark", "type": "audio"}
        ]
        
        # Check which models actually need downloading
        models_to_download = []
        for model in models:
            model_id = model["name"]
            model_info = download_status["models"][model_id]
            
            # Check if model already exists and is complete
            if model["type"] == "video":
                model_path = pathlib.Path(f"../models/video/{model_id}")
            elif model["type"] == "image":
                model_path = pathlib.Path(f"../models/image/{model_id}")
            elif model["type"] == "audio":
                model_path = pathlib.Path(f"../models/audio/{model_id}")
            
            # Check if model has actual weight files (not just config)
            has_weights = False
            if model_path.exists():
                # Look for actual model weight files
                weight_files = list(model_path.rglob("*.safetensors")) + list(model_path.rglob("*.bin")) + list(model_path.rglob("*.pt")) + list(model_path.rglob("*.pth"))
                has_weights = len(weight_files) > 0
            
            if has_weights:
                # Model already exists and is complete
                download_status["models"][model_id].update({
                    "downloaded_mb": model_info["size_gb"] * 1024,
                    "progress": 100,
                    "speed_mbps": 0,
                    "eta_seconds": 0,
                    "status": "completed"
                })
                print(f"✅ Model {model_id} already exists and is complete")
            else:
                # Model needs downloading
                models_to_download.append(model)
                print(f"📥 Model {model_id} needs downloading")
        
        if not models_to_download:
            # All models already exist
            download_status.update({
                "is_downloading": False,
                "progress": 100,
                "current_model": "",
                "status": "completed",
                "message": "All models already downloaded!",
                "error": None
            })
            print("All models already exist, no download needed")
            return
        
        total_models = len(models_to_download)
        
        for i, model in enumerate(models_to_download):
            model_id = model["name"]
            model_info = download_status["models"][model_id]
            
            # Update current model status
            download_status.update({
                "current_model": model["name"],
                "message": f"Downloading {model['name']}...",
                "progress": int((i / total_models) * 100)
            })
            
            # Set model status to downloading
            download_status["models"][model_id]["status"] = "downloading"
            
            print(f"🚀 Starting download of {model['name']}...")
            
            # Use Python script directly with proper virtual environment
            venv_python = pathlib.Path(__file__).parent / "venv" / "bin" / "python"
            if venv_python.exists():
                python_cmd = str(venv_python)
            else:
                python_cmd = "python3"
            
            # Run download script for specific model
            result = subprocess.run(
                [python_cmd, str(script_path), "--model", model["name"]],
                capture_output=True,
                text=True,
                cwd=pathlib.Path(__file__).parent.parent
            )
            
            if result.returncode != 0:
                download_status["models"][model_id]["status"] = "error"
                print(f"❌ Failed to download {model['name']}: {result.stderr}")
                raise Exception(f"Failed to download {model['name']}: {result.stderr}")
            
            print(f"✅ Successfully downloaded {model['name']}")
            
            # Mark model as completed
            download_status["models"][model_id].update({
                "downloaded_mb": model_info["size_gb"] * 1024,
                "progress": 100,
                "speed_mbps": 0,
                "eta_seconds": 0,
                "status": "completed"
            })
        
        # Download completed
        download_status.update({
            "is_downloading": False,
            "progress": 100,
            "current_model": "",
            "status": "completed",
            "message": "All models downloaded successfully!",
            "error": None
        })
        
        # Models downloaded successfully - they will be loaded when requested
        print("Models downloaded successfully. Use the UI to load them when ready.")
        
        # Generate voice previews after successful download
        try:
            print("🎤 Generating voice previews...")
            import asyncio
            asyncio.create_task(generate_voice_previews())
        except Exception as preview_error:
            print(f"⚠️  Voice preview generation failed: {preview_error}")
        
    except Exception as e:
        download_status.update({
            "is_downloading": False,
            "status": "error",
            "message": f"Download failed: {str(e)}",
            "error": str(e)
        })

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
        job_queue.update_job(job_id, {"status": "processing", "progress": 10})
        
        # Generate video
        output_path = await video_generator.generate(
            prompt=request.prompt,
            model_name=request.model_name,
            duration=request.duration,
            output_format=request.output_format
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
