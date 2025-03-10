# YOLO Object Detection API on Kubernetes

A production-ready API for real-time object detection using YOLO models, deployed on Kubernetes with a focus on traffic monitoring and analysis.

## Features

- 🚀 FastAPI backend for image processing and model inference
- 🔍 YOLOv8 model for state-of-the-art object detection
- 🚦 Traffic monitoring specialized endpoint
- 📊 Prometheus metrics for performance monitoring
- 🔄 Kubernetes deployment with auto-scaling
- 🎯 CI/CD pipeline for automated testing and deployment

## Project Structure

The project follows a modular structure:

```
yolo-k8s-api/
├── app/                  # Main application code
│   ├── main.py           # FastAPI application entry point
│   ├── config.py         # Application configuration
│   ├── models.py         # YOLO model implementation
│   ├── schemas.py        # Pydantic data models
│   ├── middleware.py     # Custom middleware
│   ├── routes/           # API endpoints
│   └── utils/            # Utility functions
├── docker/               # Docker configuration
│   ├── app.Dockerfile    # API container definition
│   └── docker-compose.yml # Local development setup
├── k8s/                  # Kubernetes manifests
├── triton/               # Triton Server configuration
├── monitoring/           # Monitoring setup
├── tests/                # Test suite
└── docs/                 # Documentation
```

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Kubernetes cluster (for deployment)

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yolo-k8s-api.git
   cd yolo-k8s-api
   ```

2. Run with Docker Compose:
   ```bash
   cd docker
   docker-compose up
   ```

3. Access the API:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs

### API Usage

#### Basic Object Detection

```bash
curl -X POST http://localhost:8000/v1/detect \
  -F "file=@/path/to/your/image.jpg"
```

#### Traffic Analysis

```bash
curl -X POST http://localhost:8000/v1/detect/traffic \
  -F "file=@/path/to/your/traffic_image.jpg" \
  -F "visualize=true"
```

## Kubernetes Deployment

For Kubernetes deployment, apply the manifests:

```bash
# Deploy to development environment
kubectl apply -k k8s/overlays/dev

# Deploy to production environment
kubectl apply -k k8s/overlays/prod
```

## Monitoring

The API includes built-in metrics for Prometheus, accessible at `/metrics` endpoint. These can be visualized in Grafana dashboards for:

- Request rates and latencies
- Model inference times
- Detection counts by class
- Resource utilization

## Testing

Run the test suite:

```bash
cd tests
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.