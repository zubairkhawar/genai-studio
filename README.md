# GenStudio - AI Media Generation Platform

A comprehensive web application for generating high-quality videos, images, and audio using state-of-the-art AI models. Built with FastAPI backend and Next.js frontend.

## 🚀 Features

### 🎬 Video Generation
- **Stable Video Diffusion (SVD)** - Image-to-video generation
- **SD + SVD Pipeline** - Text-to-video using Stable Diffusion + SVD
- **Multiple Resolutions**: 256×256, 384×384, 512×512
- **Output Formats**: MP4, GIF
- **Cross-Platform GPU Support**: Optimized for NVIDIA GPUs (6GB-24GB+ VRAM)
- **Memory Optimized**: Automatic GPU detection and configuration

### 🖼️ Image Generation
- **Stable Diffusion v1.5** - High-quality text-to-image generation
- **Customizable Settings**: Resolution, steps, guidance scale
- **Multiple Styles**: Photorealistic, artistic, custom prompts

### 🎵 Audio Generation
- **Bark** - High-quality text-to-speech and audio generation
- **Multiple Voices**: Various speaker options
- **Audio Formats**: WAV, MP3

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework
- **Diffusers** - Hugging Face diffusion models
- **CUDA/MPS** - GPU acceleration support
- **FFmpeg** - Video processing and assembly

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icons
- **Real-time Updates** - WebSocket-like job monitoring

## 🎯 Video Generation Pipeline

```
Text Prompt → Stable Diffusion → SVD → FFmpeg → Final Video
```

1. **Stable Diffusion** generates initial keyframe image
2. **Stable Video Diffusion** creates motion from the image
3. **FFmpeg** assembles frames into final video

## 🖥️ System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060, RTX 2060, etc.)
- **RAM**: 16GB system RAM
- **Storage**: 100GB free space for models
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS (Apple Silicon)

### Recommended Requirements
- **GPU**: NVIDIA RTX 4070+ or AMD 7900 XTX (12GB+ VRAM)
- **RAM**: 32GB system RAM
- **Storage**: 200GB+ free space
- **OS**: Windows 11, Linux (Ubuntu 22.04+)

### High-End Configuration
- **GPU**: RTX 4090, RTX 6000, A100 (24GB+ VRAM)
- **RAM**: 64GB+ system RAM
- **Storage**: 500GB+ NVMe SSD

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/zubairkhawar/genai-studio.git
cd genai-studio
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend
python main.py
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

### 4. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📦 Model Management

### Download Models
1. Go to **Settings** page
2. Click **"Download AI Models"**
3. Select models to download:
   - **Stable Video Diffusion**: 32.61 GB
   - **Stable Diffusion v1.5**: 44.01 GB
   - **Bark**: ~4 GB

### Model Storage
- Models are stored in `models/` directory
- Automatic model detection and loading
- Memory-efficient loading for different GPU sizes

## 🎮 Usage Guide

### Video Generation
1. Navigate to **Video Generator**
2. Enter your text prompt
3. Select resolution (256×256, 384×384, 512×512)
4. Choose output format (MP4, GIF)
5. Click **"Generate Video"**
6. Monitor progress in **Jobs/Queue**

### Image Generation
1. Go to **Image Generator**
2. Enter descriptive prompt
3. Adjust settings (resolution, steps, guidance)
4. Generate and download

### Audio Generation
1. Visit **Audio Generator**
2. Enter text to convert to speech
3. Select voice style
4. Generate audio file

## ⚙️ GPU Optimization

### Automatic GPU Detection
The system automatically detects your GPU and applies optimal settings:

- **6GB GPUs**: 256×256, 4-6 frames, memory optimized
- **8GB GPUs**: 256×256, 8 frames, aggressive memory management
- **12GB GPUs**: 384×384, 16 frames, balanced performance
- **16GB GPUs**: 512×512, 20 frames, high performance
- **24GB+ GPUs**: 512×512, 25 frames, maximum performance

### Memory Management Features
- ✅ Automatic attention slicing
- ✅ VAE tiling for low-VRAM GPUs
- ✅ CPU offload for memory-constrained systems
- ✅ Platform-specific optimizations (Windows/Linux)
- ✅ Memory cache clearing before/after generation

## 🔧 Configuration

### Environment Variables
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0  # Use specific GPU
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Memory Management
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # For MPS
```

### Backend Configuration
Edit `backend/config.py` for custom settings:
- Model paths
- GPU memory limits
- Generation parameters

## 📊 Performance Benchmarks

### Video Generation Times (8GB GPU)
- **256×256, 8 frames**: ~2-3 minutes
- **384×384, 16 frames**: ~4-6 minutes
- **512×512, 25 frames**: ~8-12 minutes

### Memory Usage
- **SD Model**: ~4GB VRAM
- **SVD Model**: ~8-12GB VRAM (depending on resolution)
- **Total Peak**: ~15GB VRAM for 512×512 generation

## 🐛 Troubleshooting

### Common Issues

#### "CUDA out of memory"
- Reduce resolution to 256×256
- Decrease number of frames
- Enable CPU offload in settings

#### "Models not loading"
- Check if models are downloaded in Settings
- Verify GPU drivers are up to date
- Restart backend after model download

#### "Generation fails"
- Check Jobs/Queue for error messages
- Ensure sufficient disk space
- Verify internet connection for model downloads

### Performance Tips
1. **Close other GPU-intensive applications**
2. **Use SSD storage for models**
3. **Enable hardware acceleration in browser**
4. **Monitor GPU temperature and usage**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Diffusers library and model hosting
- **Stability AI** for Stable Diffusion and Stable Video Diffusion
- **Suno AI** for Bark text-to-speech model
- **FastAPI** and **Next.js** communities for excellent frameworks

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/zubairkhawar/genai-studio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zubairkhawar/genai-studio/discussions)
- **Documentation**: [Wiki](https://github.com/zubairkhawar/genai-studio/wiki)

---

**Made with ❤️ for the AI community**