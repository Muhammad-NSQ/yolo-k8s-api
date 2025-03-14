FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
#FROM nvcr.io/nvidia/onnxruntime:24.02-py3
#nvcr.io/nvidia/pytorch:24.02-py3


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    libnvinfer10 \
    libnvinfer-plugin10 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir onnxruntime-gpu==1.21.0

# Install TensorRT Python package
RUN pip install --no-cache-dir nvidia-pyindex && \
    pip install --no-cache-dir nvidia-tensorrt

WORKDIR /app

# Copy the application
COPY app/ /app/

# Create uploads and TensorRT directories
RUN mkdir -p /tmp/uploads && chmod 777 /tmp/uploads && \
    mkdir -p /tmp/tensorrt && chmod 777 /tmp/tensorrt

# Install additional requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/opt/conda/lib/python3.11/site-packages/tensorrt_libs

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]