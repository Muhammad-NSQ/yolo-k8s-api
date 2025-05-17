FROM ultralytics/ultralytics:8.3.91


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Ensure the target directory exists before running wget
RUN apt-get update -y && \
    apt-get install -y wget && \
    mkdir -p /usr/src/ultralytics/ultralytics/engine && \
    wget -O /usr/src/ultralytics/ultralytics/engine/exporter.py https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/engine/exporter.py


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

# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/opt/conda/lib/python3.11/site-packages/tensorrt_libs

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]