# SAM2 Annotator Dockerfile (GPU-enabled)

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the SAM2 repository
RUN git clone https://github.com/facebookresearch/sam2.git

# Change directory to the cloned repo
WORKDIR /workspace/sam2

# Download SAM2 Model checkpoints
RUN cd checkpoints && ./download_ckpts.sh && cd ..

# Upgrade pip and install Python requirements
RUN pip3 install --upgrade pip && pip3 install -e .

# Install PyTorch with CUDA support (correct for CUDA 12.1)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install some additional packages
RUN pip install opencv-python matplotlib 

# For annotator web app:
RUN pip install gradio pillow

WORKDIR /workspace

# Default to bash terminal
CMD ["/bin/bash"]
