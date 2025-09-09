# GenAI Media Studio 🎬🎵

A modern, AI-powered desktop/web application for generating video and audio from text prompts. Features a beautiful, futuristic UI with dark/light mode support and runs entirely locally with support for both NVIDIA and AMD GPUs.

![GenAI Media Studio](https://img.shields.io/badge/GenAI-Media%20Studio-blue?style=for-the-badge&logo=sparkles)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Next.js](https://img.shields.io/badge/Next.js-14+-black?style=for-the-badge&logo=next.js)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3+-38B2AC?style=for-the-badge&logo=tailwind-css)

## ✨ Features

### 🎨 Modern UI/UX
- **Dark/Light Mode**: Beautiful theme switching with smooth transitions
- **Glassmorphism Design**: Modern frosted glass effects and subtle animations
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **AI-Inspired Aesthetics**: Futuristic design with glowing accents and smooth animations

### 🎬 Text-to-Video Generation
- **Stable Video Diffusion**: High-quality video generation
- **ModelScope Integration**: Chinese model support
- **Open-Sora Support**: Open-source video generation
- **Multiple Formats**: MP4, WebM, AVI output support

### 🎵 Text-to-Audio Generation
- **Bark Integration**: High-quality text-to-speech
- **RVC Support**: Voice conversion and cloning
- **AudioLDM**: Text-to-audio generation
- **Multiple Formats**: WAV, MP3, FLAC output support

### ⚡ Performance & Compatibility
- **GPU Support**: NVIDIA CUDA, AMD ROCm, Apple Silicon MPS
- **Local Processing**: No cloud dependencies, complete privacy
- **Job Queue**: Asynchronous generation with progress tracking
- **FFmpeg Integration**: Professional media conversion

### 🛠️ Developer Features
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Real-time Updates**: WebSocket-like job status updates
- **Model Management**: Load/unload models via web UI
- **Cross-platform**: Windows, macOS, Linux support

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- FFmpeg
- GPU with CUDA/ROCm support (optional, CPU fallback available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zubairkhawar/genai-studio.git
   cd genai-studio
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Access the Application**
   - Open http://localhost:3000 in your browser
   - Backend API runs on http://localhost:8000

### One-Click Setup
```bash
# Cross-platform setup script
./setup.sh  # Linux/macOS
setup.bat   # Windows
```

**Quick Start:**
```bash
# Clone and setup
git clone https://github.com/zubairkhawar/genai-studio.git
cd genai-studio

# Run setup (choose your platform)
./setup.sh      # Linux/macOS
# OR
setup.bat       # Windows

# Start the application
./start.sh      # Linux/macOS
# OR
start.bat       # Windows
```

## 🎯 Usage

### Basic Workflow
1. **Choose Media Type**: Select Video or Audio generation
2. **Select Model**: Pick from available AI models
3. **Enter Prompt**: Describe what you want to create
4. **Configure Settings**: Adjust duration, quality, format (optional)
5. **Generate**: Click the magic button and watch it create!
6. **Preview & Download**: View results and download files

### Advanced Features
- **Model Management**: Load/unload models to optimize memory usage
- **Job Queue**: Monitor multiple generations simultaneously
- **Progress Tracking**: Real-time progress updates with beautiful animations
- **Batch Processing**: Queue multiple jobs for efficient workflow

## 🏗️ Architecture

### Backend (FastAPI)
```
backend/
├── main.py                 # FastAPI application
├── models/
│   ├── video_generator.py  # Video generation logic
│   └── audio_generator.py  # Audio generation logic
├── utils/
│   ├── gpu_detector.py     # GPU detection & configuration
│   ├── ffmpeg_handler.py   # Media conversion utilities
│   └── job_queue.py        # Asynchronous job management
└── requirements.txt        # Python dependencies
```

### Frontend (Next.js + TailwindCSS)
```
frontend/
├── src/
│   ├── app/
│   │   └── page.tsx        # Main application page
│   ├── components/
│   │   ├── GenerationForm.tsx    # Input form
│   │   ├── JobQueue.tsx          # Job management
│   │   ├── MediaPreview.tsx      # Results preview
│   │   ├── GPUInfo.tsx           # System info
│   │   └── ModelManager.tsx      # Model controls
│   ├── contexts/
│   │   └── ThemeContext.tsx      # Theme management
│   └── hooks/
│       └── useThemeColors.ts     # Theme utilities
├── tailwind.config.ts      # Tailwind configuration
└── package.json           # Node.js dependencies
```

## 🎨 Theme System

### Dark Mode (Default)
- **Primary Background**: Deep slate navy (#0f172a)
- **Secondary Panels**: Lighter slate (#1e293b)
- **Accent Colors**: Neon blue (#38bdf8), violet (#a78bfa), lime green (#84cc16)
- **Glass Effects**: Subtle transparency with backdrop blur

### Light Mode
- **Primary Background**: Almost white (#f8fafc)
- **Secondary Panels**: Soft gray (#e2e8f0)
- **Accent Colors**: Bright blue (#0284c7), vibrant violet (#7c3aed), fresh green (#65a30d)
- **Glass Effects**: Clean white cards with soft shadows

## 🔧 Configuration

### GPU Detection
The application automatically detects and configures:
- **NVIDIA GPUs**: CUDA support with automatic memory detection
- **AMD GPUs**: ROCm support with memory management
- **Apple Silicon**: MPS acceleration for M1/M2/M3 chips
- **CPU Fallback**: Automatic fallback for systems without GPU

### Model Management
- **Dynamic Loading**: Load models on-demand to save memory
- **Model Switching**: Easy switching between different models
- **Memory Optimization**: Automatic cleanup of unused models

## 📱 Mobile Support

The application is fully responsive with:
- **Collapsible Panels**: Optimized for mobile screens
- **Touch-Friendly**: Large buttons and touch targets
- **Adaptive Layout**: Stack layout on smaller screens
- **Gesture Support**: Swipe and touch interactions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stability AI** for Stable Video Diffusion
- **Suno AI** for Bark text-to-speech
- **Hugging Face** for AudioLDM and model hosting
- **FastAPI** team for the excellent web framework
- **Next.js** team for the React framework
- **TailwindCSS** for the utility-first CSS framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/zubairkhawar/genai-studio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zubairkhawar/genai-studio/discussions)
- **Email**: [Your Email]

## 🗺️ Roadmap

- [x] **One-Click Installer**: Cross-platform setup scripts ✅
- [x] **Modern UI/UX**: Dark/light mode with glassmorphism design ✅
- [x] **GPU Support**: NVIDIA CUDA, AMD ROCm, Apple Silicon MPS ✅
- [x] **Job Queue System**: Asynchronous generation with progress tracking ✅
- [x] **FFmpeg Integration**: Professional media conversion ✅
- [x] **Model Management**: Load/unload models via web UI ✅
- [ ] **Docker Support**: Containerized deployment
- [ ] **Model Marketplace**: Easy model installation
- [ ] **Batch Processing**: Multiple prompt processing
- [ ] **API Documentation**: Interactive API docs
- [ ] **Plugin System**: Extensible architecture
- [ ] **Cloud Sync**: Optional cloud storage integration
- [ ] **Mobile App**: Native mobile applications

---

**Made with ❤️ and AI magic by [Your Name]**

*Transform your ideas into reality with the power of AI*
