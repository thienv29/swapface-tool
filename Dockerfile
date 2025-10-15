# Use NVIDIA CUDA base image for GTX 1650 support
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# Set environment variables for GPU
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Optimize apt installs - consolidate into single layer and clean up cache
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata \
    ffmpeg \
    libcudnn8=8.6.0.163-1+cuda11.8 \
    libcudnn8-dev=8.6.0.163-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy requirements first for better caching (layer optimization)
COPY requirements.txt .

# Optimize pip installs: use --no-cache-dir to reduce layer size
RUN pip3 install --no-cache-dir --compile --progress-bar off -r requirements.txt

# Create necessary directories in same layer as code copy
RUN mkdir -p output models

# Copy application code (after pip install to maximize cache usage)
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python3", "app.py"]
