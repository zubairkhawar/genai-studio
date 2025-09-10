# Model Setup Guide

This guide explains how to download and set up the correct model weights for the Text-to-Media App to work with local models instead of making API calls.

## Overview

The application is designed to be **local-first** and uses only two high-quality models:

- **🎥 Video**: `stabilityai/stable-video-diffusion-img2vid` - Official SVD for image-to-video generation
- **🎙️ Audio**: `suno/bark` - High-quality text-to-speech generation

Both models are optimized for local inference and provide excellent results without requiring API calls.

## Quick Start

### Option 1: Download All Models (Recommended)

```bash
# Make the script executable (if not already)
chmod +x scripts/download-models.sh

# Download all available models
./scripts/download-models.sh --all
```

### Option 2: Download Specific Models

```bash
# Download only video models
./scripts/download-models.sh --video

# Download only audio models
./scripts/download-models.sh --audio

# Download a specific model
./scripts/download-models.sh --model stable-video-diffusion
./scripts/download-models.sh --model bark
```

### Option 3: Using Python Script Directly

```bash
# Download all models
python3 scripts/download-models.py --all

# Download specific model
python3 scripts/download-models.py --model bark

# List available models
python3 scripts/download-models.py --list
```

## Available Models

### Video Model

| Model | Repository | Size | Type | Description |
|-------|------------|------|------|-------------|
| `stable-video-diffusion` | `stabilityai/stable-video-diffusion-img2vid` | ~5GB | Image-to-Video | Official SVD - High-quality video generation from images |

**Features:**
- Takes an input image + text prompt → generates short video clip
- Actively maintained by Stability AI
- Runs on CUDA, ROCm (AMD), and MPS (Mac M1/M2)
- Outputs 2–4 sec videos at 576×1024 resolution
- Includes frame interpolation for smoother results

### Audio Model

| Model | Repository | Size | Type | Description |
|-------|------------|------|------|-------------|
| `bark` | `suno/bark` | ~4GB | Text-to-Speech | High-quality text-to-speech and audio generation |

**Features:**
- Converts text → realistic speech/audio
- Supports multiple voices and non-speech sounds
- Lightweight enough for Mac M1/M2 or CPU-only inference
- Works offline, no API required
- Supports various audio formats (WAV, MP3)

## Directory Structure

After downloading, your models will be organized as follows:

```
models/
├── video/
│   └── stable-video-diffusion/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── scheduler/
│       ├── transformer/
│       └── vae/
└── audio/
    └── bark/
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer/
```

## How It Works

### Local Loading Priority

The application follows this priority order when loading models:

1. **Local Directory**: Check if model exists in `/models/` directory
2. **Hugging Face Download**: Download from Hugging Face if local model not found
3. **Placeholder**: Use lightweight placeholder if both fail

### Model Loading Process

When you start the application:

1. The backend checks for local model weights in the `/models/` directory
2. If found, it loads the model from local files (faster, offline)
3. If not found, it downloads from Hugging Face (slower, requires internet)
4. If download fails, it uses a placeholder for demo purposes

## Requirements

### System Requirements

- **Disk Space**: ~15GB for all models
- **RAM**: 8GB+ recommended for video models
- **GPU**: CUDA-compatible GPU recommended (CPU fallback available)

### Dependencies

The download script will automatically install required dependencies:

- `huggingface_hub` - For downloading models from Hugging Face
- `torch` - PyTorch for model inference
- `diffusers` - Hugging Face Diffusers library
- `bark` - Bark text-to-speech library

## Troubleshooting

### Common Issues

#### 1. "Model not found" Error

```bash
# Check if model was downloaded correctly
ls -la models/video/stable-video-diffusion/
ls -la models/audio/bark/

# Re-download if needed
./scripts/download-models.sh --model stable-video-diffusion --force
```

#### 2. "Out of memory" Error

- Reduce batch size in model configuration
- Use CPU instead of GPU: Set `CUDA_VISIBLE_DEVICES=""`
- Download smaller models first

#### 3. "Permission denied" Error

```bash
# Make scripts executable
chmod +x scripts/download-models.sh
chmod +x scripts/download-models.py
```

#### 4. "Network error" During Download

- Check internet connection
- Try downloading models individually
- Use `--force` flag to retry failed downloads

### Verification

To verify models are loaded correctly:

1. Start the backend server
2. Check the console output for model loading messages
3. Look for "✅ Successfully loaded [model] from local weights"
4. Test generation in the web interface

## Advanced Configuration

### Custom Model Paths

You can modify the model paths in the generator files:

- `backend/models/video_generator.py` - Video model paths
- `backend/models/audio_generator.py` - Audio model paths

### Environment Variables

Set these environment variables to customize model loading:

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Set Bark cache directory
export BARK_CACHE_DIR=/path/to/bark/models

# Disable model downloads (local only)
export LOCAL_MODELS_ONLY=true
```

### Model Configuration

Edit model configurations in the generator files to:

- Change default models
- Add new model types
- Modify model parameters
- Set custom paths

## Performance Tips

### For Better Performance

1. **Use Local Models**: Always download models locally for faster loading
2. **GPU Acceleration**: Ensure CUDA is properly installed
3. **Memory Management**: Unload unused models to free memory
4. **Batch Processing**: Process multiple requests together when possible

### For Lower Resource Usage

1. **Start with Audio Models**: Audio models are smaller and faster
2. **Use Placeholder Mode**: For testing without heavy models
3. **CPU Mode**: Use CPU inference for lower memory usage
4. **Smaller Models**: Use modelscope instead of stable-video-diffusion

## Support

If you encounter issues:

1. Check the console output for error messages
2. Verify model files are complete in the `/models/` directory
3. Try re-downloading models with `--force` flag
4. Check system requirements and dependencies

## Next Steps

After setting up models:

1. Start the backend server: `cd backend && python main.py`
2. Start the frontend: `cd frontend && npm run dev`
3. Open the web interface and test model loading
4. Generate your first video or audio content!

---

**Note**: The first time you run the application, it may take a few minutes to download and cache models. Subsequent runs will be much faster with local models.
