# Use NVIDIA CUDA 12.8 base image
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build dependencies
# Since base image is bare, we need to install python explicitly
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python -> python3 if it doesn't exist
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements
COPY requirements.txt .

# Install python dependencies with increased timeout and specific index
RUN pip3 install --no-cache-dir --timeout=1000 -r requirements.txt

# Copy the rest of the application
COPY . .

# Set PYTHONPATH to include the project root
ENV PYTHONPATH=/app

CMD ["/bin/bash"]
