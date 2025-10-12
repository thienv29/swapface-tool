# SwapFace API Tool

This is a tool API for swapping faces in images or videos while preserving the original expression from the source image/video.

## Features

- Face detection using InsightFace
- Face swapping with expression preservation using landmark-based warping
- Support for both images and videos
- Flask API endpoint for easy integration

## Requirements

See requirements.txt for dependencies.

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the API server

```bash
python swapface_api.py
```

The server will start on `http://localhost:5000` by default.

#### Web Interface

Visit `http://localhost:5000` in your browser to access the web interface with:
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
python client.py --source video.mp4 --target face.jpg --server http://192.168.1.100:5000

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
curl -X POST -F "source=@source.jpg" -F "target=@target.jpg" http://localhost:5000/swapface
```

For video:
```bash
curl -X POST -F "source=@source.mp4" -F "target=@target.jpg" http://localhost:5000/swapface
```

## How it works

1. Detects faces in both source and target using InsightFace
2. Uses facial landmarks to warp the target face shape while preserving the source expression
3. Applies seamless cloning for natural blending
4. For videos: processes each frame individually

## Note

The expression preservation is achieved through affine transformation using key facial landmarks. For best results, ensure good lighting and frontal face angles.
