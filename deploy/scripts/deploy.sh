#!/bin/bash
set -e

# TEMPL Pipeline - Simple Deployment Script
# Optimized for daily development workflow
# Replaces complex build/deploy scripts with simple commands

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
step() { echo -e "${BLUE}🔄 $1${NC}"; }

# Configuration
REGISTRY="cerit.io"
DEFAULT_USERNAME=$(whoami)
DEFAULT_NAMESPACE="default"
DEFAULT_VERSION="latest"

# Help function
show_help() {
    cat << 'EOF'
TEMPL Pipeline Deployment - Simple Commands

USAGE:
  ./deploy.sh COMMAND [OPTIONS]

COMMANDS:
  build         Build Docker image
  build-init    Build data initialization container
  deploy        Full deployment to Kubernetes (with data setup)
  update        Update running deployment
  config        Update configuration only (fast, no rebuild)
  status        Check deployment status
  logs          Show application logs
  shell         Get shell access to running pod
  data-status   Check dataset status and storage usage

OPTIONS:
  -u, --username USERNAME    Harbor username (default: current user)
  -n, --namespace NAMESPACE  Kubernetes namespace (default: default)
  -v, --version VERSION      Image version (default: latest)
  -d, --domain DOMAIN        Domain name (without .dyn.cloud.e-infra.cz)
  --push                     Push image to Harbor after build
  --help                     Show this help

EXAMPLES:
  # Build image
  ./deploy.sh build -u myuser --push

  # Deploy to Kubernetes
  ./deploy.sh deploy -u myuser -n mynamespace -d mydomain

  # Update running deployment
  ./deploy.sh update -u myuser -n mynamespace

  # Update configuration only (fast)
  ./deploy.sh config -n mynamespace

  # Check status
  ./deploy.sh status -n mynamespace

WORKFLOW:
  1. First time: ./deploy.sh deploy -u USER -n NAMESPACE -d DOMAIN
  2. Code changes: ./quick-update.sh (30 seconds, no rebuild)
  3. Major updates: ./deploy.sh update -u USER -n NAMESPACE
  4. Check status: ./deploy.sh status -n NAMESPACE

EOF
}

# Validate prerequisites
check_prerequisites() {
    local cmd="$1"
    
    # Always check kubectl for Kubernetes operations
    if [[ "$cmd" != "build" ]]; then
        # Test kubectl with a simple command that works with limited permissions
        if ! kubectl config current-context >/dev/null 2>&1; then
            error "kubectl not configured or cluster not accessible"
            exit 1
        fi
    fi
    
    # Check Docker for build operations
    if [[ "$cmd" == "build" ]] || [[ "$cmd" == "deploy" ]] || [[ "$cmd" == "update" ]]; then
        if ! docker info >/dev/null 2>&1; then
            error "Docker not running or not accessible"
            exit 1
        fi
    fi
}

# Build Docker image
build_image() {
    step "Building Docker image"
    local image_tag="${REGISTRY}/${USERNAME}/templ-pipeline:${VERSION}-production"
    info "Image: $image_tag"
    
    if [[ ! -f "deploy/docker/Dockerfile" ]]; then
        error "deploy/docker/Dockerfile not found"
        exit 1
    fi
    
    docker build -f deploy/docker/Dockerfile -t "$image_tag" .
    success "Image built: $image_tag"
    
    if [[ "$PUSH_IMAGE" == "true" ]]; then
        step "Pushing image to registry"
        docker push "$image_tag"
        success "Image pushed: $image_tag"
    fi
}

# Deploy to Kubernetes
deploy_k8s() {
    step "Deploying to Kubernetes"
    
    local image_tag="${REGISTRY}/${USERNAME}/templ-pipeline:${VERSION}-production"
    local full_domain="${DOMAIN}.dyn.cloud.e-infra.cz"
    
    info "Configuration:"
    echo "  Username: $USERNAME"
    echo "  Namespace: $NAMESPACE"
    echo "  Domain: $full_domain"
    echo "  Image: $image_tag"
    
    # Create namespace if needed
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        step "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Update manifests
    step "Updating deployment configuration"
    
    # Create temporary manifests with substitutions
    mkdir -p /tmp/k8s-deploy
    
    # Use optimized manifests if available, fallback to original
    local deployment_file="deploy/kubernetes/deployment.yaml"
    local service_file="deploy/kubernetes/service.yaml"
    local ingress_file="deploy/kubernetes/ingress.yaml"
    local configmap_file="deploy/kubernetes/configmap.yaml"
    
    # Prefer optimized versions
    if [[ -f "deploy/kubernetes/deployment.optimized.yaml" ]]; then
        deployment_file="deploy/kubernetes/deployment.optimized.yaml"
        info "Using optimized deployment configuration"
    fi
    if [[ -f "deploy/kubernetes/service.optimized.yaml" ]]; then
        service_file="deploy/kubernetes/service.optimized.yaml"
        info "Using optimized service configuration"
    fi
    if [[ -f "deploy/kubernetes/ingress.optimized.yaml" ]]; then
        ingress_file="deploy/kubernetes/ingress.optimized.yaml"
        info "Using optimized ingress configuration"
    fi
    if [[ -f "deploy/kubernetes/configmap.optimized.yaml" ]]; then
        configmap_file="deploy/kubernetes/configmap.optimized.yaml"
        info "Using optimized configmap configuration"
    fi
    
    # Update deployment with correct image
    sed "s|cerit.io/xfulop/templ-pipeline:latest-production|${image_tag}|g; s|cerit.io/xfulop/templ-pipeline@sha256:.*|${image_tag}|g" \
        "$deployment_file" > /tmp/k8s-deploy/deployment.yaml
    
    # Update ingress with correct domain
    sed "s|templ\\.dyn\\.cloud\\.e-infra\\.cz|${full_domain}|g" \
        "$ingress_file" > /tmp/k8s-deploy/ingress.yaml
    
    # Update configmap namespace
    sed "s|namespace: default|namespace: ${NAMESPACE}|g" \
        "$configmap_file" > /tmp/k8s-deploy/configmap.yaml
    
    # Copy service manifest with correct filename
    info "Using service file: $service_file"
    cp "$service_file" /tmp/k8s-deploy/service.yaml
    
    # Validate all required files exist
    local required_files=("deployment.yaml" "service.yaml" "ingress.yaml" "configmap.yaml")
    for file in "${required_files[@]}"; do
        if [[ ! -f "/tmp/k8s-deploy/$file" ]]; then
            error "Required Kubernetes manifest missing: /tmp/k8s-deploy/$file"
            rm -rf /tmp/k8s-deploy
            exit 1
        fi
    done
    info "All Kubernetes manifests prepared successfully"
    
    # Apply resources in correct order
    step "Applying Kubernetes resources"
    
    kubectl apply -f /tmp/k8s-deploy/configmap.yaml -n "$NAMESPACE"
    kubectl apply -f /tmp/k8s-deploy/service.yaml -n "$NAMESPACE"
    
    # Apply deployment and ingress (no PVC wait needed)
    kubectl apply -f /tmp/k8s-deploy/deployment.yaml -n "$NAMESPACE"
    kubectl apply -f /tmp/k8s-deploy/ingress.yaml -n "$NAMESPACE"
    
    # Wait for deployment
    step "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/templ-pipeline -n "$NAMESPACE"
    
    # Cleanup
    rm -rf /tmp/k8s-deploy
    
    success "Deployment completed successfully!"
    info "Application available at: https://${full_domain}"
    info "Use './deploy.sh status -n $NAMESPACE' to check status"
}

# Update existing deployment
update_deployment() {
    step "Updating existing deployment"
    
    local image_tag="${REGISTRY}/${USERNAME}/templ-pipeline:${VERSION}-production"
    
    # Build new image
    build_image
    
    if [[ "$PUSH_IMAGE" == "true" ]]; then
        # Update deployment with new image
        step "Updating deployment with new image"
        kubectl set image deployment/templ-pipeline \
            templ-pipeline="$image_tag" -n "$NAMESPACE"
        
        # Wait for rollout
        kubectl rollout status deployment/templ-pipeline -n "$NAMESPACE" --timeout=300s
        
        success "Deployment updated successfully!"
    else
        warning "Image not pushed. Use --push to update deployment"
    fi
}

# Show deployment status
show_status() {
    step "Checking deployment status in namespace: $NAMESPACE"
    
    echo ""
    echo "=== PODS ==="
    kubectl get pods -l app=templ-pipeline -n "$NAMESPACE" -o wide
    
    echo ""
    echo "=== SERVICES ==="
    kubectl get svc -l app=templ-pipeline -n "$NAMESPACE"
    
    echo ""
    echo "=== INGRESS ==="
    kubectl get ingress -n "$NAMESPACE"
    
    echo ""
    echo "=== USEFUL COMMANDS ==="
    info "View logs:      ./deploy.sh logs -n $NAMESPACE"
    info "Shell access:   ./deploy.sh shell -n $NAMESPACE"
    info "Port forward:   kubectl port-forward svc/templ-pipeline-svc 8501:80 -n $NAMESPACE"
}

# Show logs
show_logs() {
    step "Showing application logs"
    kubectl logs -f deployment/templ-pipeline -n "$NAMESPACE"
}

# Update configuration only (fast)
update_config() {
    step "Updating configuration (fast, no rebuild)"
    
    info "This updates ConfigMap and restarts deployment without rebuilding Docker image"
    info "Takes ~30 seconds vs 20+ minutes for full rebuild"
    
    # Apply ConfigMap (prefer optimized version)
    step "Applying updated ConfigMap"
    local configmap_file="deploy/kubernetes/configmap.yaml"
    if [[ -f "deploy/kubernetes/configmap.optimized.yaml" ]]; then
        configmap_file="deploy/kubernetes/configmap.optimized.yaml"
        info "Using optimized ConfigMap"
    fi
    
    # Update namespace and apply
    sed "s|namespace: default|namespace: ${NAMESPACE}|g" "$configmap_file" | kubectl apply -f - -n "$NAMESPACE"
    
    # Restart deployment to pick up new config
    step "Restarting deployment to pick up new configuration"
    kubectl rollout restart deployment/templ-pipeline -n "$NAMESPACE"
    
    # Wait for rollout
    step "Waiting for rollout to complete"
    kubectl rollout status deployment/templ-pipeline -n "$NAMESPACE" --timeout=120s
    
    success "Configuration updated successfully!"
    
    # Show verification
    local pod_name=$(kubectl get pods -l app=templ-pipeline -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$pod_name" ]]; then
        info "Verification - Pod: $pod_name"
        info "Use './deploy.sh logs -n $NAMESPACE' to check if application started correctly"
    fi
    
    echo ""
    success "Benefits achieved:"
    echo "  ⚡ 30 seconds vs 20+ minutes for Docker rebuild"
    echo "  🔧 Changed configuration without code changes"
    echo "  🔄 No image rebuild required"
    echo "  📝 Version controlled configuration"
}

# Get shell access
get_shell() {
    step "Opening shell in running pod"
    kubectl exec -it deployment/templ-pipeline -n "$NAMESPACE" -- bash
}

# Build init container for data management
build_init_container() {
    step "Building data initialization container"
    
    local init_image="${REGISTRY}/${USERNAME}/templ-pipeline-data-init:latest"
    info "Building init container: $init_image"
    
    docker build -f deploy/docker/init-data.Dockerfile -t "$init_image" .
    success "Init container built: $init_image"
    
    if [[ "$PUSH_IMAGE" == "true" ]]; then
        step "Pushing init container to registry"
        docker push "$init_image"
        success "Init container pushed: $init_image"
    else
        info "To push init container: docker push $init_image"
    fi
}

# Check data status and storage usage
check_data_status() {
    step "Checking dataset status and storage usage"
    
    info "PVC Status:"
    kubectl get pvc templ-data-pvc -n "$NAMESPACE" 2>/dev/null || {
        warning "PVC templ-data-pvc not found in namespace $NAMESPACE"
        return 1
    }
    
    echo ""
    info "Data Directory Contents:"
    kubectl exec deployment/templ-pipeline -n "$NAMESPACE" -- ls -la /app/data/ 2>/dev/null || {
        warning "Could not access /app/data in running pod"
        return 1
    }
    
    echo ""
    info "Storage Usage:"
    kubectl exec deployment/templ-pipeline -n "$NAMESPACE" -- du -sh /app/data/* 2>/dev/null | sort -hr || true
    
    echo ""
    info "Disk Space:"
    kubectl exec deployment/templ-pipeline -n "$NAMESPACE" -- df -h /app/data 2>/dev/null || true
    
    echo ""
    info "Required Datasets Check:"
    kubectl exec deployment/templ-pipeline -n "$NAMESPACE" -- bash -c "
        echo -n 'Protein embeddings: '
        [[ -f /app/data/embeddings/templ_protein_embeddings_v1.0.0.npz ]] && echo '✅ Present' || echo '❌ Missing'
        
        echo -n 'Processed ligands: '
        [[ -f /app/data/ligands/templ_processed_ligands_v1.0.0.sdf.gz ]] && echo '✅ Present' || echo '❌ Missing'
        
        echo -n 'PDBBind refined: '
        [[ -d /app/data/PDBBind/PDBbind_v2020_refined ]] && echo '✅ Present' || echo '❌ Missing'
        
        echo -n 'PDBBind other_PL: '
        [[ -d /app/data/PDBBind/PDBbind_v2020_other_PL ]] && echo '✅ Present' || echo '❌ Missing'
    " 2>/dev/null || warning "Could not check dataset files"
    
    success "Data status check completed"
}

# Parse arguments
parse_args() {
    COMMAND=""
    USERNAME="$DEFAULT_USERNAME"
    NAMESPACE="$DEFAULT_NAMESPACE"
    VERSION="$DEFAULT_VERSION"
    DOMAIN=""
    PUSH_IMAGE="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            build|deploy|update|config|status|logs|shell|build-init|data-status)
                COMMAND="$1"
                shift
                ;;
            -u|--username)
                USERNAME="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -d|--domain)
                DOMAIN="$2"
                shift 2
                ;;
            --push)
                PUSH_IMAGE="true"
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
    
    # Validate required parameters
    if [[ -z "$COMMAND" ]]; then
        error "Command required"
        show_help
        exit 1
    fi
    
    if [[ "$COMMAND" == "deploy" ]] && [[ -z "$DOMAIN" ]]; then
        error "Domain required for deployment (use -d option)"
        exit 1
    fi
}

# Main function
main() {
    parse_args "$@"
    check_prerequisites "$COMMAND"
    
    case "$COMMAND" in
        build)
            build_image
            ;;
        deploy)
            if [[ "$PUSH_IMAGE" != "true" ]]; then
                PUSH_IMAGE="true"  # Auto-enable push for deployment
            fi
            build_image
            deploy_k8s
            ;;
        update)
            if [[ "$PUSH_IMAGE" != "true" ]]; then
                PUSH_IMAGE="true"  # Auto-enable push for updates
            fi
            update_deployment
            ;;
        config)
            update_config
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        shell)
            get_shell
            ;;
        build-init)
            build_init_container
            ;;
        data-status)
            check_data_status
            ;;
        *)
            error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"