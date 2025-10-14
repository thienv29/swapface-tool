# Use NVIDIA CUDA base image for GTX 1650 support
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# Set environment variables for GPU
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and pip
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
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN for GTX 1650 compatibility
RUN apt-get update && apt-get install -y \
    libcudnn8=8.6.0.163-1+cuda11.8 \
    libcudnn8-dev=8.6.0.163-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p output models

# Expose port
EXPOSE 8000

# Run the application
CMD ["python3", "app.py"]
