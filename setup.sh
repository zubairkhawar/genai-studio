#!/bin/bash

# GenAI Media Studio Setup Script
# Cross-platform setup for Linux and macOS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║              🎬 GenAI Media Studio Setup 🎵                  ║"
echo "║                                                              ║"
echo "║         AI-Powered Text-to-Video & Audio Generation          ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS"

# Check system requirements
print_status "Checking system requirements..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 1 ]]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3.8+ not found. Please install Python first."
    exit 1
fi

# Check Node.js version
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [[ $NODE_VERSION -ge 18 ]]; then
        print_success "Node.js $(node --version) found"
    else
        print_error "Node.js 18+ required, found $(node --version)"
        exit 1
    fi
else
    print_error "Node.js 18+ not found. Please install Node.js first."
    exit 1
fi

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    print_success "FFmpeg found: $(ffmpeg -version | head -n1)"
else
    print_warning "FFmpeg not found. Installing..."
    
    if [[ "$OS" == "linux" ]]; then
        # Detect Linux distribution
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            sudo pacman -S ffmpeg
        else
            print_error "Could not detect package manager. Please install FFmpeg manually."
            exit 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            print_error "Homebrew not found. Please install FFmpeg manually or install Homebrew first."
            exit 1
        fi
    fi
fi

# Check GPU support
print_status "Checking GPU support..."

if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)"
    GPU_TYPE="nvidia"
elif command -v rocm-smi &> /dev/null; then
    print_success "AMD GPU detected: $(rocm-smi --showproductname | grep "Card series" | cut -d: -f2 | xargs)"
    GPU_TYPE="amd"
else
    print_warning "No GPU detected, will use CPU (slower generation)"
    GPU_TYPE="cpu"
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p outputs/videos
mkdir -p outputs/audio
mkdir -p models
mkdir -p logs

# Install Python dependencies
print_status "Installing Python dependencies..."
cd backend

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
print_status "Installing Python packages..."
pip install -r requirements.txt

# Install PyTorch with appropriate backend
if [[ "$GPU_TYPE" == "nvidia" ]]; then
    print_status "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$GPU_TYPE" == "amd" ]]; then
    print_status "Installing PyTorch with ROCm support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
else
    print_status "Installing PyTorch with CPU support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

deactivate
cd ..

# Install Node.js dependencies
print_status "Installing Node.js dependencies..."
cd frontend
npm install
cd ..

# Install root dependencies
print_status "Installing root dependencies..."
npm install

# Create environment file
print_status "Creating environment configuration..."
cat > .env << EOF
# GenAI Media Studio Configuration
NODE_ENV=development
BACKEND_PORT=8000
FRONTEND_PORT=3000
OUTPUT_DIR=outputs
MODELS_DIR=models
GPU_TYPE=$GPU_TYPE

# Optional: Set custom model cache directory
# HF_HOME=./models/huggingface
# TRANSFORMERS_CACHE=./models/transformers
EOF

# Create startup script
print_status "Creating startup script..."
cat > start.sh << 'EOF'
#!/bin/bash

# GenAI Media Studio Startup Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting GenAI Media Studio...${NC}"

# Check if virtual environment exists
if [[ ! -d "backend/venv" ]]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source backend/venv/bin/activate

# Start backend in background
echo -e "${GREEN}Starting backend server...${NC}"
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo -e "${GREEN}Starting frontend server...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}GenAI Media Studio is running!${NC}"
echo -e "${BLUE}Frontend: http://localhost:3000${NC}"
echo -e "${BLUE}Backend API: http://localhost:8000${NC}"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${BLUE}Stopping servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    deactivate
    echo -e "${GREEN}Servers stopped.${NC}"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
EOF

chmod +x start.sh

# Create stop script
print_status "Creating stop script..."
cat > stop.sh << 'EOF'
#!/bin/bash

# GenAI Media Studio Stop Script

echo "Stopping GenAI Media Studio..."

# Kill processes by port
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

echo "GenAI Media Studio stopped."
EOF

chmod +x stop.sh

# Create update script
print_status "Creating update script..."
cat > update.sh << 'EOF'
#!/bin/bash

# GenAI Media Studio Update Script

echo "Updating GenAI Media Studio..."

# Update Python dependencies
cd backend
source venv/bin/activate
pip install --upgrade -r requirements.txt
deactivate
cd ..

# Update Node.js dependencies
cd frontend
npm update
cd ..

# Update root dependencies
npm update

echo "Update complete!"
EOF

chmod +x update.sh

# Final setup
print_status "Setting up permissions..."
chmod +x setup.sh

# Create desktop entry (Linux only)
if [[ "$OS" == "linux" ]]; then
    print_status "Creating desktop entry..."
    cat > ~/.local/share/applications/genai-media-studio.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=GenAI Media Studio
Comment=AI-Powered Text-to-Video & Audio Generation
Exec=$(pwd)/start.sh
Icon=$(pwd)/public/icon.png
Terminal=false
Categories=AudioVideo;Multimedia;
EOF
fi

# Summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete! 🎉                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
print_success "GenAI Media Studio has been successfully installed!"
echo ""
echo -e "${CYAN}Quick Start:${NC}"
echo -e "  ${YELLOW}./start.sh${NC}     - Start the application"
echo -e "  ${YELLOW}./stop.sh${NC}      - Stop the application"
echo -e "  ${YELLOW}./update.sh${NC}    - Update dependencies"
echo ""
echo -e "${CYAN}Access URLs:${NC}"
echo -e "  ${BLUE}Frontend:${NC} http://localhost:3000"
echo -e "  ${BLUE}Backend API:${NC} http://localhost:8000"
echo -e "  ${BLUE}API Docs:${NC} http://localhost:8000/docs"
echo ""
echo -e "${CYAN}GPU Support:${NC} $GPU_TYPE"
echo -e "${CYAN}Output Directory:${NC} ./outputs/"
echo -e "${CYAN}Models Directory:${NC} ./models/"
echo ""
print_warning "Note: First-time model downloads may take several minutes."
print_warning "Make sure you have a stable internet connection."
echo ""
echo -e "${PURPLE}Happy creating! 🎬🎵${NC}"