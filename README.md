# SwapFace API Tool

This is a tool API for swapping faces in images while preserving the original expression from the source image. Built with Flask, InsightFace, and ONNX Runtime for face swapping functionality.

## Features

- Face detection using InsightFace with Buffalo L model
- Real-time face swapping with expression preservation
- Support for image-to-image face swapping
- Web interface with upload and preview
- Flask REST API for easy integration
- Local model storage (no downloads required)

## Requirements

- Python 3.10+
- macOS (M1/M2 chipset supported)
- Git
- Virtual environment (venv)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/thienv29/swapface-tool.git
cd swapface-tool
```

### 2. Create and activate virtual environment
```bash
# Tạo virtual environment
python3 -m venv venv

# Activate trên macOS/Linux
source venv/bin/activate

# Activate trên Windows
# venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify model presence
The face swapping model (`models/inswapper_128.onnx`) is already included in the repository.

## Usage

### 1. Run the API server

```bash
python app.py
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
