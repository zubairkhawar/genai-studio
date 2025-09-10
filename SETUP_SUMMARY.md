# Text-to-Media App - Setup Summary

## 🎯 **Local-First Model Configuration**

Your application is now configured to use only **two high-quality, local-first models**:

### 🎥 **Video Generation**
- **Model**: `stabilityai/stable-video-diffusion-img2vid`
- **Type**: Image-to-Video (SVD)
- **Size**: ~5GB
- **Features**: 
  - Official Stable Video Diffusion from Stability AI
  - Takes input image + text prompt → generates 2-4 second videos
  - 576×1024 resolution output
  - Frame interpolation for smoother results
  - Runs on CUDA, ROCm (AMD), and MPS (Mac M1/M2)

### 🎙️ **Audio Generation**
- **Model**: `suno/bark`
- **Type**: Text-to-Speech (TTS)
- **Size**: ~4GB
- **Features**:
  - High-quality text-to-speech generation
  - Multiple voices and non-speech sounds
  - Lightweight enough for Mac M1/M2 or CPU-only inference
  - Works completely offline
  - Supports WAV, MP3 formats

## 🚀 **Quick Setup**

### 1. Download All Models (Recommended)
```bash
./scripts/download-models.sh --all
```

### 2. Download Individual Models
```bash
# Video model only
./scripts/download-models.sh --model stable-video-diffusion

# Audio model only  
./scripts/download-models.sh --model bark
```

### 3. Check Model Status
```bash
./scripts/download-models.sh --list
```

## 📁 **Directory Structure**
```
models/
├── video/
│   └── stable-video-diffusion/     # SVD model files
└── audio/
    └── bark/                       # Bark model files
```

## ⚡ **Key Benefits**

✅ **Local-First**: No API calls, works completely offline  
✅ **High Quality**: Uses the best available open-source models  
✅ **Optimized**: Only downloads what you need (9GB total)  
✅ **Cross-Platform**: Works on CUDA, ROCm, MPS, and CPU  
✅ **Reproducible**: Pinned model versions for consistent results  

## 🔧 **Technical Details**

- **Total Storage**: ~9GB for both models
- **Memory**: 8GB+ RAM recommended for video generation
- **GPU**: CUDA/ROCm/MPS recommended but not required
- **Dependencies**: Automatically managed via virtual environment

## 🎬 **Usage**

1. **Start Backend**: `cd backend && python main.py`
2. **Start Frontend**: `cd frontend && npm run dev`
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Generate Content**: 
   - Upload an image + text prompt for video generation
   - Enter text prompt for audio generation

## 📋 **Model Loading Priority**

1. **Local Models**: Loads from `/models/` directory (fastest)
2. **Hugging Face**: Downloads on first use if local not found
3. **Placeholder**: Lightweight demo mode if models unavailable

## 🛠️ **Troubleshooting**

- **Low Disk Space**: Models require ~9GB free space
- **Memory Issues**: Use CPU mode or reduce batch size
- **Download Fails**: Check internet connection, retry with `--force`
- **Model Not Loading**: Verify files in `/models/` directory

---

**Ready to generate amazing content locally! 🎉**
