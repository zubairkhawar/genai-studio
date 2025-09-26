#!/bin/bash

# Model Download Script for Text-to-Media App
# This script provides an easy way to download model weights from Hugging Face

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}🚀 Text-to-Media App - Model Downloader${NC}"
echo "================================================"

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

# Check for virtual environment
VENV_PYTHON="$PROJECT_ROOT/backend/venv/bin/python"
if [ -f "$VENV_PYTHON" ]; then
    print_info "Using virtual environment: $VENV_PYTHON"
    PYTHON_CMD="$VENV_PYTHON"
else
    print_info "Using system Python"
    PYTHON_CMD="python3"
fi

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/backend/main.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all                    Download all available models"
    echo "  --video                  Download all video models"
    echo "  --audio                  Download all audio models"
    echo "  --model MODEL_NAME       Download specific model"
    echo "  --list                   List available models"
    echo "  --force                  Force re-download even if model exists"
    echo "  --help                   Show this help message"
    echo ""
    echo "Available models:"
    echo "  Image:"
    echo "    - stable-diffusion        (Stable Diffusion v1.5 - Text-to-Image - ~4GB)"
    echo "  Audio:"
    echo "    - bark                    (Bark TTS - High-quality speech - ~4GB)"
    echo ""
    echo "Examples:"
    echo "  $0 --all                  # Download all models"
    echo "  $0 --video                # Download all video models"
    echo "  $0 --model bark           # Download only Bark model"
    echo "  $0 --list                 # List available models"
}

# Function to check disk space
check_disk_space() {
    local required_gb=15  # Approximate space needed for all models
    local available_gb=$(df -g . | awk 'NR==2 {print $4}')
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        print_warning "Low disk space detected. Available: ${available_gb}GB, Required: ~${required_gb}GB"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to download models
download_models() {
    local args="$1"
    
    print_info "Starting model download process..."
    print_info "This may take a while depending on your internet connection."
    echo ""
    
    # Check disk space before downloading
    check_disk_space
    
    # Run the Python download script
    cd "$PROJECT_ROOT"
    $PYTHON_CMD scripts/download-models.py $args
    
    if [ $? -eq 0 ]; then
        print_status "Model download completed successfully!"
        echo ""
        print_info "Models are now available in the /models/ directory"
        print_info "You can now run the application with local model weights"
    else
        print_error "Model download failed"
        exit 1
    fi
}

# Parse command line arguments
FORCE=""
MODEL=""
ACTION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ACTION="all"
            shift
            ;;
        --video)
            ACTION="video"
            shift
            ;;
        --audio)
            ACTION="audio"
            shift
            ;;
        --model)
            MODEL="$2"
            ACTION="model"
            shift 2
            ;;
        --list)
            ACTION="list"
            shift
            ;;
        --force)
            FORCE="--force"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# If no action specified, show usage
if [ -z "$ACTION" ]; then
    show_usage
    exit 0
fi

# Execute the appropriate action
case $ACTION in
    "all")
        download_models "--all $FORCE"
        ;;
    "video")
        print_info "No video models available (SVD removed)"
        ;;
    "image")
        print_info "Downloading image model..."
        download_models "--model stable-diffusion $FORCE"
        ;;
    "audio")
        print_info "Downloading audio model..."
        download_models "--model bark $FORCE"
        ;;
    "model")
        if [ -z "$MODEL" ]; then
            print_error "Model name required when using --model"
            exit 1
        fi
        download_models "--model $MODEL $FORCE"
        ;;
    "list")
        $PYTHON_CMD scripts/download-models.py --list
        ;;
esac
