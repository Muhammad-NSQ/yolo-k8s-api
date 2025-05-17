# Use the NVIDIA Triton Server base image
FROM nvcr.io/nvidia/tritonserver:23.02-py3

# Install any additional dependencies if needed
RUN pip install opencv-python && \
    apt update && \
    apt install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /model_repository

# Set default command (can be overridden in docker-compose)
CMD ["tritonserver", "--model-repository=/model_repository"]
