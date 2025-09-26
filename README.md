# Text-to-Media App

A comprehensive AI-powered application for generating videos, images, and audio from text prompts. Built with modern web technologies and optimized for local deployment.

## 🚀 Features

### 🎥 Video Generation
- **AnimateDiff** - Perfect for GIF generation with looping animations
- **Kandinsky 2.2** - Artistic image generation with unique style

### 🎵 Audio Generation
- **XTTS-v2** - High-quality multi-speaker text-to-speech
- **Bark** - Creative voice generation with multiple voices

### 🖼️ Image Generation
- **Stable Diffusion v1.5** - High-quality text-to-image generation
- **Kandinsky 2.2** - Alternative artistic style

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework
- **Diffusers** - Hugging Face diffusion models
- **TTS** - Text-to-speech library
- **OpenCV** - Video processing

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS
- **Lucide React** - Beautiful icons

## 📋 Prerequisites

- Python 3.8+
- Node.js 18+
- 8GB+ RAM (16GB+ recommended)
- GPU with 6GB+ VRAM (optional, CPU fallback available)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd text-to-media-app
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Download Models
```bash
cd backend
python scripts/download-models.py --priority
```

### 5. Start the Application
```bash
# Terminal 1 - Backend
cd backend
python main.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

Visit `http://localhost:3000` to access the application.

## 📦 Model Management

### Download Models
The application includes a comprehensive model download system:

```bash
# Download all models (~97GB)
python scripts/download-models.py --all

# Download priority models only (~62GB)
python scripts/download-models.py --priority

# Download specific model
python scripts/download-models.py --model animatediff
```

### Available Models
- **AnimateDiff** (~2GB) - GIF generation
- **Kandinsky 2.2** (~12GB) - Artistic images
- **XTTS-v2** (~4GB) - High-quality TTS
- **Stable Diffusion** (~44GB) - Text-to-image
- **Bark** (~5GB) - Creative TTS

## 🎯 Usage

### Video Generation
1. Navigate to the Video page
2. Enter your text prompt
3. Select AI model (AnimateDiff recommended for GIFs)
4. Choose output format (GIF for AnimateDiff, MP4 for others)
5. Click "Generate Video"

### Audio Generation
1. Navigate to the Audio page
2. Enter your text prompt
3. Select AI model (XTTS-v2 recommended)
4. Choose voice style and format
5. Click "Generate Audio"

### Settings Management
1. Navigate to Settings page
2. Download models using the download buttons
3. Monitor system information
4. Clear outputs to free space

## 🔧 Configuration

### Backend Configuration
Edit `backend/main.py` to modify:
- Model paths
- Download settings
- API endpoints

### Frontend Configuration
Edit `frontend/src/config.ts` to modify:
- API URLs
- Default settings

## 📁 Project Structure

```
text-to-media-app/
├── backend/
│   ├── models/           # AI model implementations
│   ├── utils/            # Utility functions
│   ├── scripts/          # Download and setup scripts
│   └── main.py          # FastAPI application
├── frontend/
│   ├── src/
│   │   ├── app/         # Next.js pages
│   │   ├── components/  # React components
│   │   └── hooks/       # Custom hooks
│   └── package.json
├── models/              # Downloaded AI models
├── outputs/             # Generated media files
└── README.md
```

## 🎨 Model Recommendations

### For GIF Generation
- **AnimateDiff** - Best for short, looping animations
- **Kandinsky** - For artistic, stylized GIFs

### For High-Quality Videos
- **AnimateDiff** - Good balance of quality and speed

### For Audio Generation
- **XTTS-v2** - Best quality, multiple speakers
- **Bark** - Creative voices, non-speech sounds

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce model batch size
   - Use CPU fallback
   - Close other applications

2. **Model Download Fails**
   - Check internet connection
   - Ensure sufficient disk space
   - Try downloading individual models

3. **Generation Fails**
   - Check model is downloaded
   - Verify GPU drivers
   - Check system requirements

### Performance Optimization

1. **GPU Acceleration**
   - Install CUDA drivers
   - Use MPS on Apple Silicon
   - Enable ROCm for AMD GPUs

2. **Memory Management**
   - Use model offloading
   - Enable attention slicing
   - Monitor memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for model hosting
- Stability AI for Stable Diffusion models
- Coqui for XTTS-v2
- Suno for Bark
- The open-source AI community

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Note**: This application is designed for local deployment and requires significant computational resources. Ensure your system meets the minimum requirements before installation.