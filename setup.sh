#!/bin/bash

# GenStudio - AI Text-to-Media Generation Platform
# Setup Script for Client Delivery

echo "🎬 GenStudio Setup Script"
echo "========================="
echo ""

# Check if Python 3.8+ is installed
echo "📋 Checking system requirements..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION found"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi

NODE_VERSION=$(node --version)
echo "✅ Node.js $NODE_VERSION found"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm."
    exit 1
fi

NPM_VERSION=$(npm --version)
echo "✅ npm $NPM_VERSION found"

echo ""
echo "🔧 Setting up backend..."

# Create virtual environment for backend
cd backend
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Backend setup complete!"
echo ""

echo "🎨 Setting up frontend..."
cd ../frontend

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete!"
echo ""

echo "📁 Creating necessary directories..."
cd ..
mkdir -p outputs/videos
mkdir -p outputs/audio
mkdir -p outputs/voice-previews
mkdir -p outputs/custom-voices
mkdir -p models/video/animatediff
mkdir -p models/audio/bark
mkdir -p models/image/stable-diffusion
mkdir -p temp

echo "✅ Directory structure created!"
echo ""

echo "🚀 Setup complete!"
echo ""
echo "To start the application:"
echo "1. Backend: cd backend && source venv/bin/activate && python main.py"
echo "2. Frontend: cd frontend && npm run dev"
echo ""
echo "The application will be available at:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo ""
echo "📖 For detailed instructions, see README.md"
