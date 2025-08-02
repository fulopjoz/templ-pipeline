#!/bin/bash
set -e

echo "======================================================="
echo "TEMPL Pipeline CERIT Deployment - Complete Setup"
echo "======================================================="

# Configuration
HARBOR_USERNAME=""
NAMESPACE=""
DOMAIN_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--username)
            HARBOR_USERNAME="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -u <harbor-username> -n <namespace> -d <domain-name>"
            echo ""
            echo "Options:"
            echo "  -u, --username    Harbor username for image registry"
            echo "  -n, --namespace   Kubernetes namespace for deployment"
            echo "  -d, --domain      Domain name for ingress (without .dyn.cloud.e-infra.cz)"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -u johndoe -n johndoe-ns -d templ-johndoe"
            echo ""
            echo "This will deploy to: https://templ-johndoe.dyn.cloud.e-infra.cz"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$HARBOR_USERNAME" ] || [ -z "$NAMESPACE" ] || [ -z "$DOMAIN_NAME" ]; then
    echo "Error: Missing required parameters"
    echo "Use: $0 -u <harbor-username> -n <namespace> -d <domain-name>"
    echo "Use: $0 -h for help"
    exit 1
fi

FULL_DOMAIN="${DOMAIN_NAME}.dyn.cloud.e-infra.cz"

echo "Configuration:"
echo "  Harbor Username: $HARBOR_USERNAME"
echo "  Namespace: $NAMESPACE"
echo "  Domain: $FULL_DOMAIN"
echo ""

read -p "Continue with deployment? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "Step 1: Updating configuration files..."

# Update ingress domain
sed -i "s/templ-pipeline\.dyn\.cloud\.e-infra\.cz/${FULL_DOMAIN}/g" deploy/kubernetes/ingress.yaml

# Update deployment image
sed -i "s|cerit.io/\[your-username\]/templ-pipeline:latest|cerit.io/${HARBOR_USERNAME}/templ-pipeline:latest|g" deploy/kubernetes/deployment.yaml

echo "✓ Configuration files updated"

echo ""
echo "Step 2: Building Docker image..."
./deploy/scripts/build.sh latest "$HARBOR_USERNAME"

echo ""
echo "Step 3: Pushing to Harbor registry..."
echo "Please ensure you are logged in to Harbor:"
echo "  docker login cerit.io"
echo ""
read -p "Press Enter to continue after logging in..."

docker push "cerit.io/${HARBOR_USERNAME}/templ-pipeline:latest"
echo "✓ Image pushed to Harbor"

echo ""
echo "Step 4: Creating Kubernetes resources..."
kubectl apply -f deploy/kubernetes/pvc.yaml -n "$NAMESPACE"

echo "Waiting for PVC to be bound..."
kubectl wait --for=condition=Bound pvc/templ-data-pvc -n "$NAMESPACE" --timeout=120s
echo "✓ PVC created and bound"

echo ""
echo "Step 5: Copying data to persistent volume..."
./deploy/scripts/copy-data.sh "$NAMESPACE"
echo "✓ Data copied to PVC"

echo ""
echo "Step 6: Deploying application..."
kubectl apply -f deploy/kubernetes/deployment.yaml -n "$NAMESPACE"
kubectl apply -f deploy/kubernetes/service.yaml -n "$NAMESPACE"
kubectl apply -f deploy/kubernetes/ingress.yaml -n "$NAMESPACE"

echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/templ-pipeline -n "$NAMESPACE"
echo "✓ Application deployed"

echo ""
echo "======================================================="
echo "DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo "======================================================="
echo ""
echo "Your TEMPL Pipeline is now available at:"
echo "  https://${FULL_DOMAIN}"
echo ""
echo "It may take a few minutes for the SSL certificate to be issued"
echo "and the DNS to propagate."
echo ""
echo "Useful commands:"
echo "  View logs:    kubectl logs -f deployment/templ-pipeline -n $NAMESPACE"
echo "  View status:  kubectl get pods,svc,ingress -l app=templ-pipeline -n $NAMESPACE"
echo "  Port forward: kubectl port-forward svc/templ-pipeline-svc 8501:80 -n $NAMESPACE"
echo ""
echo "To update the deployment:"
echo "  ./deploy/scripts/build.sh v1.1.0 $HARBOR_USERNAME true"
echo "  ./deploy/scripts/deploy-master.sh image -u $HARBOR_USERNAME -n $NAMESPACE -v v1.1.0 --push"
echo "  Or use: ./deploy/scripts/deploy-master.sh config -n $NAMESPACE  # for config-only updates"
echo ""
echo "======================================================="
