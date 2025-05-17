# Use NVIDIA TensorRT base image (same as you were using)
FROM nvcr.io/nvidia/tensorrt:23.02-py3

# Python environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install libgl1 -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (same ones you were installing manually)
RUN pip install --no-cache-dir \
    ultralytics \              
    onnxruntime-gpu \         
    onnx \
    tritonclient[all]        

# Create necessary directories
RUN mkdir -p /app /model_repository /tmp/conversion

# Copy the conversion script
COPY scripts/convert_model.py /app/

# Set working directory
WORKDIR /app

# Default command - runs the conversion script
CMD ["python", "convert_model.py"]