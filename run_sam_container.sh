#!/bin/bash

# Name the image and container
IMAGE_NAME=sam2_annotator
CONTAINER_NAME=sam2_annotator_container

# Build the Docker image
docker build -t $IMAGE_NAME .

# Path to your local scripts folder (adjust this)
LOCAL_SCRIPT_DIR=$(realpath ./scripts)
LOCAL_CONFIG_DIR=$(realpath ./config)

# Path to local data folder to mount into container for labelling
# The first command line argument will be used instead of the default when provided
DEFAULT_LOCAL_DATA_DIR="/home/csrobot/Desktop/ATB1_BENCHMARK"
LOCAL_DATA_DIR="${1:-$DEFAULT_LOCAL_DATA_DIR}"
echo "Local Data Directory: $LOCAL_DATA_DIR"

xhost +local:docker

# Run the container with GPU access and volume mounting
docker run -it --rm \
  --gpus all \
  --net=host \
  --name $CONTAINER_NAME \
  -v "$LOCAL_SCRIPT_DIR":/workspace/scripts \
  -v "$LOCAL_CONFIG_DIR":/workspace/config \
  -v "$LOCAL_DATA_DIR":/workspace/data \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -e DISPLAY=$DISPLAY \
  $IMAGE_NAME \
  /bin/bash
