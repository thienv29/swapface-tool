# SwapFace Tool - GTX 1650 Optimized

This is a tool API optimized for Windows GTX 1650 GPU for swapping faces in images and videos while preserving the original expression from the source image. Built with Flask, InsightFace, and CUDA ONNX Runtime for high-performance face swapping functionality.

**Optimized for GTX 1650 (4GB VRAM) with CUDA support**

## Features

- Face detection using InsightFace with Buffalo L model
- Real-time face swapping with expression preservation
- Support for image-to-image face swapping
- Web interface with upload and preview
- Flask REST API for easy integration
- Local model storage (no downloads required)

## Requirements

- Windows 10/11 with NVIDIA GTX 1650 GPU
- NVIDIA GPU drivers (latest)
- Docker Desktop with NVIDIA Docker support
- Docker Compose
- Git

## Installation & Docker Build (Windows)

### Prerequisites
1. **Install NVIDIA GPU Drivers**: Latest drivers for GTX 1650
2. **Install Docker Desktop**: https://www.docker.com/products/docker-desktop
3. **Enable NVIDIA Container Toolkit**:
   ```bash
   # Download and install NVIDIA Docker
   # https://github.com/NVIDIA/nvidia-docker
   ```

### 1. Clone the repository
```bash
git clone https://github.com/thienv29/swapface-tool.git
cd swapface-tool
```

### 2. Build Docker Image
```bash
# Double-click build.bat OR run manually:
docker build -t swapface-gtx1650 .
```

### 3. Verify model presence
The face swapping model (`models/inswapper_128.onnx`) is already included in the repository.

## Usage with Docker

### 1. Build the Docker Image
```bash
# Use the provided build script
build.bat

# OR build manually
docker build -t swapface-gtx1650 .
```

### 2. Run the Container
```bash
# Use the provided run script (recommended)
run.bat

# OR run manually
docker run --gpus all --name swapface-gtx1650-app -p 8000:8000 \
  -v "%CD%/output:/app/output" \
  -v "%CD%/models:/app/models" \
  -v "%CD%/static:/app/static" \
  -v "%CD%/templates:/app/templates" \
  -e CUDA_VISIBLE_DEVICES=0 \
  swapface-gtx1650
```

### 3. Alternative: Use Docker Compose
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d --build
```

The server will start on `http://localhost:8000` by default.

#### Web Interface

Visit `http://localhost:8000` in your browser to access the web interface with:
- Two tabs: "Swap Image" and "Swap Video"
- File upload for source (image/video) and target face
- Live preview of uploaded files
- Processing progress indicator
- Result display with server file paths

### 2. Use the provided client

#### Install client dependencies (if not already done globally)
```bash
pip install requests==2.32.3
```

#### Usage
```bash
python client.py --source <source_file> --target <target_file> [--server <url>] [--output-dir <dir>]
```

#### Examples
```bash
# Basic usage with default server
python client.py --source image.jpg --target face.jpg

# Specify server URL
python client.py --source video.mp4 --target face.jpg --server http://192.168.1.100:8000

# Save output to specific directory (note: output is saved on server)
python client.py --source image.png --target face.jpeg --output-dir ./results
```

### 3. Direct API usage

**POST** `/swapface`

Upload two files:
- `source`: The source image or video file
- `target`: The target face image

#### Request
- Content-Type: `multipart/form-data`
- Files: `source` and `target`

#### Response
- Success: `{"result": "<path_to_output>", "type": "image|video"}`
- Error: `{"error": "error message"}`

#### Example usage with curl

For image:
```bash
curl -X POST -F "source=@source.jpg" -F "target=@target.jpg" http://localhost:8000/swapface
```

For video:
```bash
curl -X POST -F "source=@source.mp4" -F "target=@target.jpg" http://localhost:8000/swapface
```

## How it works

1. Detects faces in both source and target using InsightFace
2. Uses facial landmarks to warp the target face shape while preserving the source expression
3. Applies seamless cloning for natural blending
4. For videos: processes each frame individually

## Note

The expression preservation is achieved through affine transformation using key facial landmarks. For best results, ensure good lighting and frontal face angles.
