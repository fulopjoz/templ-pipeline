#!/bin/bash
set -e

# TEMPL Pipeline - Persistent Storage Deployment Script
# Deploys the application with persistent storage for code and data

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Print functions
info() { echo -e "${CYAN}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
step() { echo -e "${BLUE}🚀 $1${NC}"; }

# Configuration
DEFAULT_NAMESPACE="default"

# Help function
show_help() {
    cat << 'EOF'
TEMPL Pipeline Persistent Storage Deployment

DESCRIPTION:
  Deploys TEMPL Pipeline with persistent storage for code and data.
  This prevents pod eviction due to storage limits and ensures code updates persist.

USAGE:
  ./deploy-persistent.sh [OPTIONS]

OPTIONS:
  -n, --namespace NAMESPACE  Kubernetes namespace (default: default)
  --force                    Force recreation of PVCs (will delete existing data)
  --help                     Show this help

EXAMPLES:
  # Deploy in default namespace
  ./deploy-persistent.sh

  # Deploy in specific namespace
  ./deploy-persistent.sh -n mynamespace

  # Force recreation (WARNING: deletes existing data)
  ./deploy-persistent.sh --force

WHAT THIS DOES:
  ✅ Creates PVCs for code and data storage
  ✅ Deploys application with persistent volumes
  ✅ Prevents storage-related pod evictions
  ✅ Ensures code updates persist across restarts
  ✅ Maintains data persistence

EOF
}

# Check prerequisites
check_prerequisites() {
    if ! kubectl config current-context >/dev/null 2>&1; then
        error "kubectl not configured or cluster not accessible"
        exit 1
    fi
    
    info "Checking cluster access..."
    # Try to access the target namespace instead of cluster-info
    kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || {
        info "Namespace '$NAMESPACE' doesn't exist, will create it"
    }
    
    success "Cluster access verified"
}

# Create namespace if it doesn't exist
create_namespace() {
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        step "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
        success "Namespace created"
    else
        info "Namespace '$NAMESPACE' already exists"
    fi
}

# Apply PVCs
apply_pvcs() {
    step "Applying Persistent Volume Claims"
    
    if [[ "$FORCE" == "true" ]]; then
        warning "Force flag detected - deleting existing PVCs"
        kubectl delete pvc templ-code-pvc templ-data-pvc -n "$NAMESPACE" --ignore-not-found=true
        sleep 5
    fi
    
    info "Creating PVCs for code and data storage..."
    kubectl apply -f deploy/kubernetes/code-pvc.yaml -n "$NAMESPACE"
    
    # Wait for PVCs to be bound
    info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=bound pvc templ-code-pvc -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=bound pvc templ-data-pvc -n "$NAMESPACE" --timeout=300s
    
    success "PVCs created and bound successfully"
}

# Apply deployment
apply_deployment() {
    step "Applying persistent deployment"
    
    info "Deploying application with persistent storage..."
    kubectl apply -f deploy/kubernetes/deployment.persistent.yaml -n "$NAMESPACE"
    
    success "Deployment applied successfully"
}

# Wait for deployment to be ready
wait_for_deployment() {
    step "Waiting for deployment to be ready"
    
    info "Waiting for deployment rollout..."
    kubectl rollout status deployment/templ-pipeline -n "$NAMESPACE" --timeout=600s
    
    info "Waiting for pod to be ready..."
    kubectl wait --for=condition=ready pod -l app=templ-pipeline -n "$NAMESPACE" --timeout=300s
    
    success "Deployment is ready"
}

# Show deployment status
show_status() {
    step "Deployment Status"
    
    echo ""
    echo "📊 Pod Status:"
    kubectl get pods -l app=templ-pipeline -n "$NAMESPACE"
    
    echo ""
    echo "💾 PVC Status:"
    kubectl get pvc -n "$NAMESPACE" | grep templ
    
    echo ""
    echo "🔗 Service Status:"
    kubectl get svc -l app=templ-pipeline -n "$NAMESPACE" 2>/dev/null || echo "No services found"
    
    echo ""
    success "Deployment completed successfully!"
    echo ""
    info "Next steps:"
    echo "  • Use ./quick-update-persistent.sh for code updates"
    echo "  • View logs: kubectl logs -l app=templ-pipeline -n $NAMESPACE"
    echo "  • Access application: kubectl port-forward svc/templ-pipeline 8501:8501 -n $NAMESPACE"
    echo ""
    info "Benefits of this deployment:"
    echo "  ✅ Code updates persist across pod restarts"
    echo "  ✅ No more storage-related pod evictions"
    echo "  ✅ Data is preserved in persistent storage"
    echo "  ✅ Perfect for development workflow"
}

# Parse arguments
parse_args() {
    NAMESPACE="$DEFAULT_NAMESPACE"
    FORCE="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --force)
                FORCE="true"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Main function
main() {
    parse_args "$@"
    
    step "TEMPL Pipeline Persistent Storage Deployment"
    info "Namespace: $NAMESPACE"
    
    if [[ "$FORCE" == "true" ]]; then
        warning "Force mode enabled - existing data will be deleted!"
        read -p "Are you sure you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "Deployment cancelled"
            exit 0
        fi
    fi
    
    check_prerequisites
    create_namespace
    apply_pvcs
    apply_deployment
    wait_for_deployment
    show_status
}

main "$@" 