#!/bin/bash
set -e

# TEMPL Pipeline Full Rebuild Script
# Rebuilds Docker image with full dependencies (PyTorch, transformers) and deploys

echo "=========================================="
echo "TEMPL Pipeline Full Rebuild"
echo "Installing PyTorch, transformers, and all dependencies"
echo "=========================================="

# Parse command line arguments
VERSION="latest"
HARBOR_USERNAME=""
REGISTRY="cerit.io"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--username)
            HARBOR_USERNAME="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 -u USERNAME [-v VERSION]"
            echo "  -u, --username   Harbor username (required)"
            echo "  -v, --version    Docker image version (default: latest)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$HARBOR_USERNAME" ]]; then
    echo "Error: Harbor username is required"
    echo "Usage: $0 -u USERNAME [-v VERSION]"
    exit 1
fi

echo "Version: $VERSION"
echo "Harbor username: $HARBOR_USERNAME"
echo "Registry: $REGISTRY"
echo "=========================================="

# Clean up old images (optional - commented out to avoid accidents)
echo "Note: Skipping automatic cleanup of old images for safety"
echo "To manually clean up old images:"
echo "  docker ps -a | grep templ-pipeline | awk '{print \$1}' | xargs -r docker stop"
echo "  docker ps -a | grep templ-pipeline | awk '{print \$1}' | xargs -r docker rm"
echo "  docker images | grep templ-pipeline | grep -v latest | awk '{print \$3}' | xargs -r docker rmi"

# Build with full dependencies using registry cache
echo "Building Docker image with full dependencies..."

# Check if buildx is available for caching
if command -v docker buildx >/dev/null 2>&1; then
    echo "Using Docker Buildx with registry cache optimization..."
    
    # Create buildx builder if it doesn't exist
    docker buildx create --name templ-builder --use 2>/dev/null || docker buildx use templ-builder 2>/dev/null || true
    
    # Build with cache optimization
    docker buildx build \
        --platform linux/amd64 \
        -f deploy/docker/Dockerfile \
        -t "${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:${VERSION}" \
        -t "${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:latest" \
        --cache-from type=registry,ref="${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:buildcache" \
        --cache-to type=registry,ref="${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:buildcache",mode=max \
        --push \
        .
    
    echo "✅ Build completed with registry cache optimization!"
else
    echo "Docker Buildx not available, using standard build..."
    docker build \
        -f deploy/docker/Dockerfile \
        -t "${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:${VERSION}" \
        -t "${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:latest" \
        .
    
    echo "Build completed successfully!"
    
    # Push to Harbor (only needed for standard build)
    echo "Pushing image to Harbor registry..."
    docker push "${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:${VERSION}"
    docker push "${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:latest"
    echo "✅ Image pushed successfully!"
fi

# Verify the build
echo "Verifying container with full dependencies..."
if docker run --rm "${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:${VERSION}" production python -c "
import torch
import transformers
import streamlit
import templ_pipeline
import rdkit
print('✅ All dependencies available')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print('✅ Container verification passed')
"; then
    echo "Container verification successful"
else
    echo "Container verification failed"
    exit 1
fi

# Restart Kubernetes deployment
echo "Restarting Kubernetes deployment..."
kubectl rollout restart deployment templ-pipeline

echo "Waiting for rollout to complete..."
kubectl rollout status deployment templ-pipeline --timeout=300s

# Verify deployment
echo "Verifying deployment..."
kubectl get pods -l app=templ-pipeline

echo ""
echo "=========================================="
echo "✅ Full rebuild completed successfully!"
echo "=========================================="
echo ""
echo "Testing commands:"
echo "  View logs:      kubectl logs -f deployment/templ-pipeline"
echo "  Health check:   kubectl exec deployment/templ-pipeline -- python -c 'import streamlit; import templ_pipeline; print(\"Health OK\")'"
echo "  Test PyTorch:   kubectl exec deployment/templ-pipeline -- python -c 'import torch; print(\"PyTorch:\", torch.__version__)'"
echo "  Test RDKit:     kubectl exec deployment/templ-pipeline -- python -c 'from rdkit import Chem; print(\"RDKit:\", Chem.MolFromSmiles(\"CCO\") is not None)'"
echo ""
echo "Next steps:"
echo "  ./deploy/scripts/deploy-master.sh status  # Check deployment status"
echo "  ./deploy/scripts/update-config.sh        # Update configuration only"
echo "==========================================" 