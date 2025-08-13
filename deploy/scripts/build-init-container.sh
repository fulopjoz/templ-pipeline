#!/bin/bash
set -e

# Build script for TEMPL Pipeline data initialization container

REGISTRY="cerit.io"
USERNAME="xfulop"
IMAGE_NAME="templ-pipeline-data-init"
VERSION="latest"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO] $1${NC}"; }
success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
warning() { echo -e "${YELLOW}[WARNING] $1${NC}"; }

echo "=========================================="
echo "Building TEMPL Pipeline Init Container"
echo "=========================================="

FULL_IMAGE_NAME="${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION}"

info "Building init container image..."
info "Image: $FULL_IMAGE_NAME"
info "Context: ."
info "Dockerfile: deploy/docker/init-data.Dockerfile"

# Build the image
docker build \
    -f deploy/docker/init-data.Dockerfile \
    -t "$FULL_IMAGE_NAME" \
    .

success "Init container built successfully: $FULL_IMAGE_NAME"

# Ask if user wants to push
echo ""
read -p "Push to registry? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Pushing to registry..."
    docker push "$FULL_IMAGE_NAME"
    success "Image pushed successfully"
else
    warning "Image not pushed. To push later:"
    echo "  docker push $FULL_IMAGE_NAME"
fi

echo ""
success "Init container build completed!"
info "To use in deployment:"
echo "  image: $FULL_IMAGE_NAME"