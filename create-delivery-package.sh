#!/bin/bash

# GenStudio - Client Delivery Package Creator
# This script creates a clean delivery package for the client

echo "📦 Creating GenStudio Delivery Package..."
echo "========================================"
echo ""

# Create delivery directory
DELIVERY_DIR="genai-studio-delivery"
echo "📁 Creating delivery directory: $DELIVERY_DIR"
rm -rf $DELIVERY_DIR
mkdir -p $DELIVERY_DIR

# Copy essential files
echo "📋 Copying project files..."

# Copy main directories
cp -r backend $DELIVERY_DIR/
cp -r frontend $DELIVERY_DIR/
cp -r scripts $DELIVERY_DIR/

# Copy essential files
cp README.md $DELIVERY_DIR/
cp setup.sh $DELIVERY_DIR/
cp .gitignore $DELIVERY_DIR/

# Create empty directories for outputs and models
mkdir -p $DELIVERY_DIR/outputs/videos
mkdir -p $DELIVERY_DIR/outputs/audio
mkdir -p $DELIVERY_DIR/outputs/voice-previews
mkdir -p $DELIVERY_DIR/outputs/custom-voices
mkdir -p $DELIVERY_DIR/models/video/animatediff
mkdir -p $DELIVERY_DIR/models/audio/bark
mkdir -p $DELIVERY_DIR/models/image/stable-diffusion
mkdir -p $DELIVERY_DIR/temp

# Create .gitkeep files to preserve empty directories
touch $DELIVERY_DIR/outputs/videos/.gitkeep
touch $DELIVERY_DIR/outputs/audio/.gitkeep
touch $DELIVERY_DIR/outputs/voice-previews/.gitkeep
touch $DELIVERY_DIR/outputs/custom-voices/.gitkeep
touch $DELIVERY_DIR/models/video/animatediff/.gitkeep
touch $DELIVERY_DIR/models/audio/bark/.gitkeep
touch $DELIVERY_DIR/models/image/stable-diffusion/.gitkeep
touch $DELIVERY_DIR/temp/.gitkeep

# Remove unnecessary files
echo "🧹 Cleaning up unnecessary files..."

# Remove virtual environments
rm -rf $DELIVERY_DIR/backend/venv
rm -rf $DELIVERY_DIR/backend/venv311

# Remove node_modules
rm -rf $DELIVERY_DIR/frontend/node_modules

# Remove build artifacts
rm -rf $DELIVERY_DIR/frontend/.next
rm -rf $DELIVERY_DIR/frontend/out

# Remove Python cache
find $DELIVERY_DIR -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find $DELIVERY_DIR -name "*.pyc" -delete 2>/dev/null || true

# Remove OS files
find $DELIVERY_DIR -name ".DS_Store" -delete 2>/dev/null || true

# Create client-specific files
echo "📝 Creating client-specific files..."

# Create client setup instructions
cat > $DELIVERY_DIR/CLIENT_SETUP.md << 'EOF'
# 🎬 GenStudio - Client Setup Instructions

## Quick Start

1. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

2. **Start the application:**
   
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

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## First Time Setup

1. **Download AI Models:**
   - Go to Settings → Model Management
   - Click "Download All Models" (this may take 30-60 minutes)
   - Or download models individually as needed

2. **Generate Your First Media:**
   - Try generating a simple video: "A cat walking in a garden"
   - Try generating audio: "Hello, this is a test of the text-to-speech system"
   - Try generating an image: "A beautiful sunset over mountains"

## System Requirements

- **Python 3.8+**
- **Node.js 18+**
- **8GB+ RAM** (16GB recommended)
- **20GB+ free disk space**
- **Internet connection** (for model downloads)

## Support

For technical support, refer to the main README.md file or contact the development team.

---
**GenStudio** - AI-Powered Media Generation Platform
EOF

# Create a simple start script
cat > $DELIVERY_DIR/start.sh << 'EOF'
#!/bin/bash

echo "🎬 Starting GenStudio..."
echo "======================="

# Check if setup has been run
if [ ! -d "backend/venv" ] || [ ! -d "frontend/node_modules" ]; then
    echo "⚠️  Setup not complete. Running setup first..."
    ./setup.sh
fi

echo "🚀 Starting backend..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!

echo "⏳ Waiting for backend to start..."
sleep 5

echo "🎨 Starting frontend..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ GenStudio is starting up!"
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user to stop
wait $BACKEND_PID $FRONTEND_PID
EOF

chmod +x $DELIVERY_DIR/start.sh

# Create ZIP archive
echo "📦 Creating delivery archive..."
zip -r genai-studio-delivery.zip $DELIVERY_DIR -x "*.git*" "*.DS_Store*"

# Clean up
rm -rf $DELIVERY_DIR

echo ""
echo "✅ Delivery package created successfully!"
echo ""
echo "📁 Files created:"
echo "   - genai-studio-delivery.zip (Complete delivery package)"
echo ""
echo "📋 Package contents:"
echo "   - Complete source code"
echo "   - Setup scripts"
echo "   - Documentation"
echo "   - Empty directory structure"
echo "   - Client setup instructions"
echo ""
echo "🚀 Ready for client delivery!"
echo ""
echo "To deliver to client:"
echo "1. Send genai-studio-delivery.zip"
echo "2. Client extracts and runs ./setup.sh"
echo "3. Client runs ./start.sh to launch"
