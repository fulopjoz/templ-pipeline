#!/bin/bash
set -e

# TEMPL Pipeline CERIT Deployment Script

# Parse command line arguments
NAMESPACE=""
HARBOR_USERNAME=""
VERSION="latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -u|--username)
            HARBOR_USERNAME="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 -n NAMESPACE -u USERNAME [-v VERSION]"
            echo "  -n, --namespace   Kubernetes namespace (required)"
            echo "  -u, --username    Harbor username (required)"
            echo "  -v, --version     Docker image version (default: latest)"
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
if [[ -z "$NAMESPACE" ]] || [[ -z "$HARBOR_USERNAME" ]]; then
    echo "Error: Namespace and username are required"
    echo "Usage: $0 -n NAMESPACE -u USERNAME [-v VERSION]"
    exit 1
fi

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

# Verify Kubernetes manifests exist
if [ ! -f "deploy/kubernetes/deployment.yaml" ] || [ ! -f "deploy/kubernetes/service.yaml" ] || [ ! -f "deploy/kubernetes/pvc.yaml" ] || [ ! -f "deploy/kubernetes/ingress.yaml" ]; then
    echo "Error: One or more Kubernetes manifest files not found in deploy/kubernetes/"
    exit 1
fi

# Check if namespace exists, create if it doesn't
if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
    echo "Namespace $NAMESPACE does not exist. Creating..."
    kubectl create namespace "$NAMESPACE"
fi

# Create backup and update deployment with correct image name
cp deploy/kubernetes/deployment.yaml deploy/kubernetes/deployment.yaml.bak
sed -i "s|cerit.io/\[your-username\]/templ-pipeline:latest|cerit.io/${HARBOR_USERNAME}/templ-pipeline:${VERSION}|g" deploy/kubernetes/deployment.yaml

echo "Creating/updating resources in namespace: $NAMESPACE"

# Apply Kubernetes resources
echo "Creating PersistentVolumeClaim..."
kubectl apply -f deploy/kubernetes/pvc.yaml -n "$NAMESPACE"

echo "Waiting for PVC to be bound..."
kubectl wait --for=condition=Bound pvc/templ-data-pvc -n "$NAMESPACE" --timeout=60s

echo "Creating deployment..."
kubectl apply -f deploy/kubernetes/deployment.yaml -n "$NAMESPACE"

echo "Creating service..."
kubectl apply -f deploy/kubernetes/service.yaml -n "$NAMESPACE"

echo "Creating ingress..."
kubectl apply -f deploy/kubernetes/ingress.yaml -n "$NAMESPACE"

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
echo "Application will be available at the configured ingress URL"
echo "Check ingress configuration: kubectl get ingress -n $NAMESPACE"
echo ""
echo "Useful commands:"
echo "  View logs:      kubectl logs -f deployment/templ-pipeline -n $NAMESPACE"
echo "  Describe pods:  kubectl describe pods -l app=templ-pipeline -n $NAMESPACE"
echo "  Port forward:   kubectl port-forward svc/templ-pipeline-svc 8501:80 -n $NAMESPACE"
echo "  Update config:  ./deploy/scripts/update-config.sh"
echo "  Master script:  ./deploy/scripts/deploy-master.sh --help"
echo "=========================================="
