#!/bin/bash
set -e

# TEMPL Pipeline CERIT Deployment Script

NAMESPACE=${1:-"your-namespace"}  # Replace with your actual namespace
HARBOR_USERNAME=${2:-$USER}
VERSION=${3:-latest}

echo "=========================================="
echo "Deploying TEMPL Pipeline to CERIT"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Harbor username: $HARBOR_USERNAME"
echo "Version: $VERSION"
echo "=========================================="

# Check if kubectl is configured
if ! kubectl cluster-info >/dev/null 2>&1; then
    echo "Error: kubectl is not configured or cluster is not accessible"
    echo "Please configure kubectl for CERIT cluster first"
    exit 1
fi

# Update deployment with correct image name
sed -i "s|hub.cerit.io/\[your-username\]/templ-pipeline:latest|hub.cerit.io/${HARBOR_USERNAME}/templ-pipeline:${VERSION}|g" k8s/deployment.yaml

echo "Creating/updating resources in namespace: $NAMESPACE"

# Apply Kubernetes resources
echo "Creating PersistentVolumeClaim..."
kubectl apply -f k8s/pvc.yaml -n "$NAMESPACE"

echo "Waiting for PVC to be bound..."
kubectl wait --for=condition=Bound pvc/templ-data-pvc -n "$NAMESPACE" --timeout=60s

echo "Creating deployment..."
kubectl apply -f k8s/deployment.yaml -n "$NAMESPACE"

echo "Creating service..."
kubectl apply -f k8s/service.yaml -n "$NAMESPACE"

echo "Creating ingress..."
kubectl apply -f k8s/ingress.yaml -n "$NAMESPACE"

echo "Deployment initiated! Checking status..."

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/templ-pipeline -n "$NAMESPACE"

# Show status
echo ""
echo "=========================================="
echo "Deployment Status:"
echo "=========================================="
kubectl get pods,svc,ingress -l app=templ-pipeline -n "$NAMESPACE"

echo ""
echo "=========================================="
echo "Application will be available at:"
echo "https://templ-pipeline.dyn.cloud.e-infra.cz"
echo ""
echo "To check logs:"
echo "kubectl logs -f deployment/templ-pipeline -n $NAMESPACE"
echo ""
echo "To check pod details:"
echo "kubectl describe pod -l app=templ-pipeline -n $NAMESPACE"
echo "=========================================="
