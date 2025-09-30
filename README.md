# 🎬 GenStudio - AI Text-to-Media Generation Platform

A comprehensive AI-powered platform for generating videos, audio, and images from text prompts using state-of-the-art models like AnimateDiff, Bark, and Stable Diffusion.

## ✨ Features

- **🎥 Text-to-Video**: Generate videos using AnimateDiff with customizable settings
- **🎵 Text-to-Audio**: Create speech from text using Bark TTS with multiple voice options
- **🖼️ Text-to-Image**: Generate images using Stable Diffusion
- **⚙️ Advanced Settings**: Fine-tune generation parameters for optimal results
- **📱 Modern UI**: Beautiful, responsive interface with dark/light themes
- **🔄 Real-time Progress**: Live generation progress tracking
- **📁 File Management**: Download, preview, and manage generated media
- **🎛️ Model Management**: Download and manage AI models

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **Git** (for cloning the repository)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zubairkhawar/genai-studio.git
   cd genai-studio
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

3. **Start the application:**
   
   **Backend (Terminal 1):**
   ```bash
   cd backend
   source venv/bin/activate
   python main.py
   ```
   
   **Frontend (Terminal 2):**
   ```bash
   cd frontend
   npm run dev
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## 📋 Manual Setup

If the setup script doesn't work, follow these manual steps:

### Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Directory Structure

```bash
mkdir -p outputs/videos
mkdir -p outputs/audio
mkdir -p outputs/voice-previews
mkdir -p outputs/custom-voices
mkdir -p models/video/animatediff
mkdir -p models/audio/bark
mkdir -p models/image/stable-diffusion
mkdir -p temp
```

## 🎯 Usage

### Video Generation

1. Navigate to the **Video** tab
2. Enter your text prompt
3. Adjust settings (optional):
   - Resolution (256px - 1024px)
   - Frame count (4-48 frames)
   - FPS (6-24 FPS)
   - Quality settings
4. Click **Generate Video**

### Audio Generation

1. Navigate to the **Audio** tab
2. Enter your text prompt
3. Select a voice from available options
4. Adjust audio settings if needed
5. Click **Generate Audio**

### Image Generation

1. Navigate to the **Image** tab
2. Enter your text prompt
3. Adjust image settings
4. Click **Generate Image**

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Optional: Override default paths
OUTPUTS_DIR=/path/to/outputs
MODELS_DIR=/path/to/models
TEMP_DIR=/path/to/temp

# Optional: API configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Model Management

- **Download Models**: Go to Settings → Model Management
- **Model Storage**: Models are stored in `models/` directory
- **Supported Models**:
  - AnimateDiff (Video generation)
  - Bark (Text-to-speech)
  - Stable Diffusion (Image generation)

#### Download Script Usage

The `scripts/download-models.py` script provides comprehensive model management:

```bash
# Download all models
python scripts/download-models.py --all

# Download priority models only
python scripts/download-models.py --priority

# Download specific model
python scripts/download-models.py --model bark

# List available models
python scripts/download-models.py --list

# Verify existing downloads
python scripts/download-models.py --verify

# Force re-download
python scripts/download-models.py --model bark --force
```

#### Voice Preview Generation

When downloading Bark models, the script automatically generates voice previews:

- **Automatic Generation**: Voice previews are created after successful Bark model download
- **10 English Speakers**: Generates previews for v2/en_speaker_0 through v2/en_speaker_9
- **Smart Caching**: Skips existing previews to avoid regeneration
- **Output Location**: Preview files are saved to `outputs/voice-previews/`
- **Format**: Generated as `.wav` files for optimal quality

**Example voice preview generation:**
```bash
python scripts/download-models.py --model bark
# This will:
# 1. Download Bark models
# 2. Download preset audio files
# 3. Generate voice previews automatically
# 4. Clean up non-English files
```

## 🛠️ Development

### Project Structure

```
genai-studio/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main application
│   ├── config.py           # Configuration system
│   ├── models/             # AI model implementations
│   └── utils/              # Utility functions
├── frontend/               # Next.js frontend
│   ├── src/app/           # Application pages
│   ├── src/components/    # React components
│   ├── src/contexts/      # React contexts
│   └── src/hooks/         # Custom hooks
├── scripts/               # Utility scripts
├── models/               # Downloaded AI models
├── outputs/              # Generated media files
└── setup.sh             # Setup script
```

### API Endpoints

- `GET /health` - Health check
- `POST /generate` - Generate media
- `GET /models` - List available models
- `GET /outputs/{type}` - List generated files
- `DELETE /outputs/{type}/{filename}` - Delete files
- `GET /download-status` - Model download status
- `GET /voice-previews` - List available voice previews
- `POST /generate-voice-previews` - Generate voice preview samples
- `GET /outputs/voice-previews/{filename}` - Serve voice preview files

## 🔧 Troubleshooting

### Common Issues

1. **"Module not found" errors:**
   ```bash
   cd frontend && npm install
   cd backend && pip install -r requirements.txt
   ```

2. **Port already in use:**
   - Backend: Change port in `backend/main.py`
   - Frontend: Change port in `frontend/package.json`

3. **Model download fails:**
   - Check internet connection
   - Ensure sufficient disk space
   - Try downloading models individually

4. **Generation fails:**
   - Check if models are downloaded
   - Verify GPU/CPU compatibility
   - Check system requirements

5. **Voice preview generation fails:**
   - Ensure Bark models are properly downloaded
   - Check if `soundfile` package is installed: `pip install soundfile`
   - Verify `outputs/voice-previews/` directory exists
   - Try regenerating: `python scripts/download-models.py --model bark --force`

### System Requirements

- **Minimum RAM**: 8GB
- **Recommended RAM**: 16GB+
- **Storage**: 20GB+ free space
- **GPU**: Optional but recommended for faster generation

## 📝 License

This project is proprietary software. All rights reserved.

## 🆘 Support

For technical support or questions:
- Check the troubleshooting section above
- Review the API documentation
- Contact the development team

## 🔄 Updates

To update the application:

```bash
git pull origin main
cd frontend && npm install
cd ../backend && pip install -r requirements.txt
```

---

**GenStudio** - Powered by AI, Built for Creativity 🎨