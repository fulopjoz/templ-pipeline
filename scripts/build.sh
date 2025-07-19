#!/bin/bash
set -e

# TEMPL Pipeline Docker Build Script for CERIT deployment

VERSION=${1:-latest}
HARBOR_USERNAME=${2:-$USER}
REGISTRY="hub.cerit.io"

echo "=========================================="
echo "Building TEMPL Pipeline Docker image"
echo "=========================================="
echo "Version: $VERSION"
echo "Harbor username: $HARBOR_USERNAME"
echo "Registry: $REGISTRY"
echo "=========================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or not accessible"
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build \
    -t "${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:${VERSION}" \
    -t "${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:latest" \
    .

echo "Build completed successfully!"

# Show image information
echo ""
echo "Image details:"
docker images "${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:${VERSION}"

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Login to Harbor: docker login ${REGISTRY}"
echo "2. Push image: docker push ${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:${VERSION}"
echo "3. Update k8s/deployment.yaml with your image name"
echo "4. Deploy to CERIT using kubectl"
echo "=========================================="
