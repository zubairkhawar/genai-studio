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
import uuid

from models.video_generator import VideoGenerator
from models.audio_generator import AudioGenerator
from utils.gpu_detector import GPUDetector
from utils.ffmpeg_handler import FFmpegHandler
from utils.job_queue import JobQueue

app = FastAPI(title="Text-to-Media Generator", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving generated media
app.mount("/outputs", StaticFiles(directory="../outputs"), name="outputs")

# Initialize components
gpu_detector = GPUDetector()
ffmpeg_handler = FFmpegHandler()
job_queue = JobQueue()

# Global model instances
video_generator = None
audio_generator = None

class GenerationRequest(BaseModel):
    prompt: str
    model_type: str  # "video" or "audio"
    model_name: str
    duration: Optional[int] = 5  # for video
    sample_rate: Optional[int] = 22050  # for audio
    output_format: str = "mp4"  # mp4, wav, mp3

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
    
    # Initialize generators
    video_generator = VideoGenerator(gpu_info)
    audio_generator = AudioGenerator(gpu_info)
    
    # Load default models
    await video_generator.load_default_models()
    await audio_generator.load_default_models()

@app.get("/")
async def root():
    return {"message": "Text-to-Media Generator API", "version": "1.0.0"}

@app.get("/gpu-info")
async def get_gpu_info():
    """Get GPU information"""
    return gpu_detector.detect_gpu()

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
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

@app.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job"""
    success = job_queue.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job cancelled"}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
