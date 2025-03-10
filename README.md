# YOLO Object Detection API on Kubernetes

A production-ready API for real-time object detection using YOLO models, deployed on Kubernetes.

## Features

- FastAPI backend for image processing and model inference
- YOLO model serving with Triton Inference Server
- Kubernetes deployment with auto-scaling
- Observability with Prometheus, Grafana, and centralized logging
- CI/CD pipeline for automated testing, building, and deployment

## Getting Started

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/yolo-k8s-api.git
cd yolo-k8s-api

# Run with Docker Compose
docker-compose -f docker/docker-compose.yml up
```

### Kubernetes Deployment

```bash
# Deploy to development environment
kubectl apply -k k8s/overlays/dev

# Deploy to production environment
kubectl apply -k k8s/overlays/prod
```

## Documentation

See the [docs](./docs) directory for detailed documentation.
