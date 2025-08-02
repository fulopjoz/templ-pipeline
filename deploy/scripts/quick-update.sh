#!/bin/bash
set -e

# TEMPL Pipeline - Quick Code Update Script
# Updates code in running container without Docker rebuild
# Perfect for development workflow: 30 seconds vs 20+ minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Print functions
info() { echo -e "${CYAN}[INFO] $1${NC}"; }
success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
warning() { echo -e "${YELLOW}[WARNING] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; }
step() { echo -e "${BLUE}[STEP] $1${NC}"; }

# Configuration
DEFAULT_NAMESPACE="default"

# Help function
show_help() {
    cat << 'EOF'
TEMPL Pipeline Quick Code Update

DESCRIPTION:
  Updates Python code in running container without rebuilding Docker image.
  Perfect for development - takes ~30 seconds vs 20+ minutes for full rebuild.

USAGE:
  ./quick-update.sh [OPTIONS]

OPTIONS:
  -n, --namespace NAMESPACE  Kubernetes namespace (default: default)
  --dry-run                  Show what would be updated without applying
  --help                     Show this help

EXAMPLES:
  # Quick update in default namespace
  ./quick-update.sh

  # Update in specific namespace
  ./quick-update.sh -n mynamespace

  # See what would be changed
  ./quick-update.sh --dry-run

WORKFLOW:
  1. Make changes to Python code in templ_pipeline/
  2. Run ./quick-update.sh
  3. Code updated in 30 seconds!

WHAT THIS DOES:
  ✅ Copies templ_pipeline/ code to running container
  ✅ Copies scripts/ to running container
  ✅ Restarts Streamlit app with new code
  ✅ Preserves data and configuration
  ❌ Does NOT update system dependencies (use ./deploy.sh update for that)

EOF
}

# Check prerequisites
check_prerequisites() {
    if ! kubectl config current-context >/dev/null 2>&1; then
        error "kubectl not configured or cluster not accessible"
        exit 1
    fi
    
    if ! kubectl get deployment templ-pipeline -n "$NAMESPACE" >/dev/null 2>&1; then
        error "TEMPL Pipeline deployment not found in namespace '$NAMESPACE'"
        info "Deploy first with: ./deploy.sh deploy -n $NAMESPACE -u USER -d DOMAIN"
        exit 1
    fi
    
    if ! kubectl get pods -l app=templ-pipeline -n "$NAMESPACE" | grep -q Running; then
        error "No running TEMPL Pipeline pods found in namespace '$NAMESPACE'"
        exit 1
    fi
}

# Get running pod name
get_pod_name() {
    POD_NAME=$(kubectl get pods -l app=templ-pipeline -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [[ -z "$POD_NAME" ]]; then
        error "Could not find running pod"
        exit 1
    fi
    
    info "Target pod: $POD_NAME"
}

# Copy code to container
copy_code() {
    step "Copying updated code to container"
    
    # Copy main application code
    info "Updating templ_pipeline/ code..."
    kubectl cp templ_pipeline/ "$NAMESPACE/$POD_NAME:/app/" -c templ-pipeline
    
    # Copy scripts
    if [[ -d "scripts" ]]; then
        info "Updating scripts/ code..."
        kubectl cp scripts/ "$NAMESPACE/$POD_NAME:/app/" -c templ-pipeline
    fi
    
    success "Code copied successfully"
}

# Restart application
restart_app() {
    step "Restarting Streamlit application"
    
    # Kill existing Streamlit process and restart
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -c templ-pipeline -- bash -c "
        # Kill existing streamlit processes
        pkill -f streamlit || true
        sleep 2
        
        # Start new streamlit process in background
        cd /app
        nohup streamlit run templ_pipeline/ui/app.py \
            --server.headless=true \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.fileWatcherType=none \
            > /tmp/streamlit.log 2>&1 &
        
        echo 'Streamlit restarted'
    "
    
    success "Application restarted with new code"
}

# Verify update
verify_update() {
    step "Verifying application is running"
    
    # Wait a moment for app to start
    sleep 5
    
    # Check if streamlit is responding
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -c templ-pipeline -- curl -s http://localhost:8501/_stcore/health >/dev/null
    
    if [[ $? -eq 0 ]]; then
        success "Application is responding correctly"
        info "Health check passed"
    else
        warning "Application may be starting up..."
        info "Check logs with: kubectl logs $POD_NAME -n $NAMESPACE"
    fi
}

# Show dry run
show_dry_run() {
    step "DRY RUN - What would be updated:"
    
    echo ""
    echo "Files that would be copied:"
    
    if [[ -d "templ_pipeline" ]]; then
        echo "  [OK] templ_pipeline/ → /app/templ_pipeline/"
        find templ_pipeline -name "*.py" | head -10 | sed 's/^/    /'
        local py_count=$(find templ_pipeline -name "*.py" | wc -l)
        if [[ $py_count -gt 10 ]]; then
            echo "    ... and $((py_count - 10)) more Python files"
        fi
    else
        echo "  [ERROR] templ_pipeline/ directory not found"
    fi
    
    if [[ -d "scripts" ]]; then
        echo "  [OK] scripts/ → /app/scripts/"
        find scripts -name "*.py" | head -5 | sed 's/^/    /'
    else
        echo "  [WARNING] scripts/ directory not found (optional)"
    fi
    
    echo ""
    echo "Actions that would be performed:"
    echo "  1. Copy code to pod: $POD_NAME"
    echo "  2. Kill existing Streamlit process"
    echo "  3. Start new Streamlit process with updated code"
    echo "  4. Verify application is responding"
    
    echo ""
    info "Run without --dry-run to apply these changes"
}

# Performance summary
show_summary() {
    echo ""
    step "Update completed!"
    echo ""
    success "Performance benefits:"
    echo "  - Update time: ~30 seconds (vs 20+ minutes for rebuild)"
    echo "  - No Docker rebuild required"
    echo "  - Preserves data and configuration"
    echo "  - Perfect for development workflow"
    
    echo ""
    info "Next steps:"
    echo "  - Test your changes in the application"
    echo "  - View logs: kubectl logs $POD_NAME -n $NAMESPACE"
    echo "  - For dependency changes: ./deploy.sh update -n $NAMESPACE"
}

# Parse arguments
parse_args() {
    NAMESPACE="$DEFAULT_NAMESPACE"
    DRY_RUN="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
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
    
    step "TEMPL Pipeline Quick Code Update"
    info "Namespace: $NAMESPACE"
    
    check_prerequisites
    get_pod_name
    
    if [[ "$DRY_RUN" == "true" ]]; then
        show_dry_run
        exit 0
    fi
    
    # Perform update
    copy_code
    restart_app
    verify_update
    show_summary
}

main "$@"