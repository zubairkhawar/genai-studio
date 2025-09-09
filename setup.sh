#!/bin/bash

# GenAI Media Studio - One-Click Setup Script
# Supports macOS, Linux, and Windows (via WSL)

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
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║                                                              ║"
echo "  ║    🎬 GenAI Media Studio - One-Click Setup 🎵              ║"
echo "  ║                                                              ║"
echo "  ║    Transform your ideas into reality with AI magic! ✨      ║"
echo "  ║                                                              ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
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

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    print_status "Detected OS: $OS"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python not found! Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_status "Python version: $PYTHON_VERSION"
    
    # Install pip if not available
    if ! command_exists pip3 && ! command_exists pip; then
        print_status "Installing pip..."
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        $PYTHON_CMD get-pip.py
        rm get-pip.py
    fi
    
    # Install virtual environment
    if ! command_exists virtualenv; then
        print_status "Installing virtualenv..."
        pip install virtualenv
    fi
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Install requirements
    print_status "Installing Python packages..."
    pip install --upgrade pip
    pip install -r backend/requirements.txt
    
    print_success "Python dependencies installed!"
}

# Install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    
    if ! command_exists node; then
        print_error "Node.js not found! Please install Node.js 18+ first."
        print_status "Visit: https://nodejs.org/"
        exit 1
    fi
    
    NODE_VERSION=$(node --version)
    print_status "Node.js version: $NODE_VERSION"
    
    if ! command_exists npm; then
        print_error "npm not found! Please install npm first."
        exit 1
    fi
    
    # Install frontend dependencies
    cd frontend
    npm install
    cd ..
    
    print_success "Node.js dependencies installed!"
}

# Install FFmpeg
install_ffmpeg() {
    print_status "Installing FFmpeg..."
    
    if command_exists ffmpeg; then
        print_success "FFmpeg already installed!"
        return
    fi
    
    case $OS in
        "macos")
            if command_exists brew; then
                brew install ffmpeg
            else
                print_warning "Homebrew not found. Please install FFmpeg manually:"
                print_warning "Visit: https://ffmpeg.org/download.html"
            fi
            ;;
        "linux")
            if command_exists apt-get; then
                sudo apt-get update
                sudo apt-get install -y ffmpeg
            elif command_exists yum; then
                sudo yum install -y ffmpeg
            elif command_exists dnf; then
                sudo dnf install -y ffmpeg
            else
                print_warning "Package manager not found. Please install FFmpeg manually:"
                print_warning "Visit: https://ffmpeg.org/download.html"
            fi
            ;;
        "windows")
            print_warning "Please install FFmpeg manually for Windows:"
            print_warning "Visit: https://ffmpeg.org/download.html"
            ;;
    esac
    
    print_success "FFmpeg installation completed!"
}

# Check GPU support
check_gpu_support() {
    print_status "Checking GPU support..."
    
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected!"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    elif command_exists rocm-smi; then
        print_success "AMD GPU detected!"
        rocm-smi --showproductname
    else
        print_warning "No GPU detected. The application will run on CPU."
        print_warning "For better performance, consider installing CUDA or ROCm drivers."
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p outputs/videos
    mkdir -p outputs/audio
    mkdir -p models
    print_success "Directories created!"
}

# Create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."
    
    # Backend startup script
    cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting GenAI Media Studio Backend..."
cd backend
source ../venv/bin/activate
python main.py
EOF
    chmod +x start_backend.sh
    
    # Frontend startup script
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "🎨 Starting GenAI Media Studio Frontend..."
cd frontend
npm run dev
EOF
    chmod +x start_frontend.sh
    
    # Combined startup script
    cat > start_app.sh << 'EOF'
#!/bin/bash
echo "🎬 Starting GenAI Media Studio..."
echo "Backend will run on http://localhost:8000"
echo "Frontend will run on http://localhost:3000"
echo ""
echo "Starting backend in background..."
cd backend
source ../venv/bin/activate
python main.py &
BACKEND_PID=$!

echo "Starting frontend..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo "Application started! Press Ctrl+C to stop."
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"

# Wait for user to stop
wait
EOF
    chmod +x start_app.sh
    
    print_success "Startup scripts created!"
}

# Main installation function
main() {
    print_status "Starting GenAI Media Studio setup..."
    
    # Detect OS
    detect_os
    
    # Create directories
    create_directories
    
    # Install dependencies
    install_python_deps
    install_node_deps
    install_ffmpeg
    
    # Check GPU support
    check_gpu_support
    
    # Create startup scripts
    create_startup_scripts
    
    # Final message
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}║    🎉 Setup Complete! GenAI Media Studio is ready! 🎉     ║${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    print_success "Installation completed successfully!"
    echo ""
    print_status "To start the application:"
    echo -e "  ${CYAN}./start_app.sh${NC}     - Start both backend and frontend"
    echo -e "  ${CYAN}./start_backend.sh${NC}  - Start backend only"
    echo -e "  ${CYAN}./start_frontend.sh${NC} - Start frontend only"
    echo ""
    print_status "Access the application at:"
    echo -e "  ${CYAN}Frontend:${NC} http://localhost:3000"
    echo -e "  ${CYAN}Backend API:${NC} http://localhost:8000"
    echo ""
    print_status "For more information, check the README.md file."
    echo ""
}

# Run main function
main "$@"
