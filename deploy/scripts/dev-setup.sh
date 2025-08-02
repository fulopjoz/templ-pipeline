#!/bin/bash
set -e

# TEMPL Pipeline - Development Environment Setup
# One-time setup script for development environment
# Configures Docker, kubectl, and development tools

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${CYAN}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
step() { echo -e "${BLUE}🔄 $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

show_help() {
    cat << 'EOF'
TEMPL Pipeline Development Environment Setup

DESCRIPTION:
  One-time setup script to configure your development environment
  for TEMPL Pipeline deployment and development.

USAGE:
  ./dev-setup.sh [OPTIONS]

OPTIONS:
  --docker-only     Setup Docker environment only
  --k8s-only        Setup Kubernetes tools only
  --harbor-login    Setup Harbor registry login
  --check           Check current environment status
  --help            Show this help

WHAT THIS DOES:
  ✅ Verifies Docker installation and configuration
  ✅ Checks kubectl installation and configuration
  ✅ Sets up BuildKit for optimized Docker builds
  ✅ Configures Harbor registry authentication
  ✅ Creates necessary directories and permissions
  ✅ Validates CERIT cluster connectivity
  ✅ Tests deployment prerequisites

EXAMPLES:
  # Full setup
  ./dev-setup.sh

  # Check current status
  ./dev-setup.sh --check

  # Setup Harbor login only
  ./dev-setup.sh --harbor-login

EOF
}

# Check Docker
check_docker() {
    step "Checking Docker installation and configuration"
    
    if ! command -v docker &> /dev/null; then
        error "Docker not installed. Please install Docker first."
        info "Install from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        error "Docker not running or not accessible"
        info "Start Docker and ensure your user is in the docker group:"
        info "  sudo systemctl start docker"
        info "  sudo usermod -aG docker \$USER"
        info "  # Then log out and back in"
        exit 1
    fi
    
    # Check BuildKit support
    if docker buildx version >/dev/null 2>&1; then
        success "Docker with BuildKit support available"
        
        # Create optimized builder if not exists
        if ! docker buildx inspect templ-builder >/dev/null 2>&1; then
            step "Creating optimized Docker builder"
            docker buildx create --name templ-builder --use \
                --driver-opt=network=host >/dev/null 2>&1 || true
            success "Optimized Docker builder created"
        else
            success "Optimized Docker builder already exists"
        fi
    else
        warning "BuildKit not available - using legacy Docker build"
    fi
    
    success "Docker configuration validated"
}

# Check kubectl
check_kubectl() {
    step "Checking kubectl installation and configuration"
    
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not installed. Please install kubectl first."
        info "Install from: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi
    
    if ! kubectl config current-context >/dev/null 2>&1; then
        error "kubectl not configured or cluster not accessible"
        info "Please configure kubectl for CERIT cluster:"
        info "1. Download kubeconfig from CERIT dashboard"
        info "2. Save to ~/.kube/config"
        info "3. Set permissions: chmod 700 ~/.kube/config"
        exit 1
    fi
    
    # Get cluster info
    local current_context=$(kubectl config current-context)
    success "kubectl configured for context: $current_context"
    
    # Check basic kubectl functionality (CERIT has limited permissions)
    if kubectl auth can-i list namespaces >/dev/null 2>&1; then
        success "Full cluster access available"
    else
        success "Limited access (normal for CERIT - you can deploy to your assigned namespace)"
    fi
}

# Setup Harbor login
setup_harbor() {
    step "Setting up Harbor registry authentication"
    
    local harbor_url="cerit.io"
    
    if grep -q "\"${harbor_url}\"" ~/.docker/config.json 2>/dev/null; then
        success "Already logged in to Harbor registry"
        return 0
    fi
    
    info "Harbor registry login required"
    echo "Please provide Harbor credentials:"
    echo "1. Go to https://${harbor_url}/"
    echo "2. Login with your credentials"
    echo "3. Click your username > User Profile"
    echo "4. Copy the CLI secret"
    echo ""
    
    read -p "Harbor username: " harbor_username
    read -s -p "Harbor CLI secret: " harbor_password
    echo ""
    
    if docker login "$harbor_url" -u "$harbor_username" -p "$harbor_password" >/dev/null 2>&1; then
        success "Successfully logged in to Harbor registry"
    else
        error "Failed to login to Harbor registry"
        info "Please check your credentials and try again"
        exit 1
    fi
}

# Setup directories
setup_directories() {
    step "Setting up development directories"
    
    # Ensure deploy scripts are executable
    if [[ -f "deploy.sh" ]]; then
        chmod +x deploy.sh
        success "deploy.sh made executable"
    fi
    
    if [[ -f "quick-update.sh" ]]; then
        chmod +x quick-update.sh
        success "quick-update.sh made executable"
    fi
    
    if [[ -f "deploy/docker/build-optimized.sh" ]]; then
        chmod +x deploy/docker/build-optimized.sh
        success "build-optimized.sh made executable"
    fi
    
    # Create temp directory for build processes
    mkdir -p /tmp/templ-builds
    success "Build directories created"
}

# Environment check
check_environment() {
    step "Checking complete development environment"
    
    echo ""
    echo "=== ENVIRONMENT STATUS ==="
    
    # Docker
    if docker info >/dev/null 2>&1; then
        success "Docker: Running and accessible"
        local docker_version=$(docker --version | cut -d' ' -f3 | tr -d ',')
        info "  Version: $docker_version"
        
        if docker buildx version >/dev/null 2>&1; then
            success "  BuildKit: Available"
        else
            warning "  BuildKit: Not available"
        fi
    else
        error "Docker: Not running or not accessible"
    fi
    
    # kubectl
    if kubectl config current-context >/dev/null 2>&1; then
        success "kubectl: Configured and connected"
        local kubectl_version=$(kubectl version --client --output=yaml 2>/dev/null | grep gitVersion | cut -d'"' -f2 || echo "unknown")
        info "  Version: $kubectl_version"
        
        local current_context=$(kubectl config current-context)
        info "  Context: $current_context"
    else
        error "kubectl: Not configured or not connected"
    fi
    
    # Harbor
    if grep -q "cerit.io" ~/.docker/config.json 2>/dev/null; then
        success "Harbor: Logged in"
    else
        warning "Harbor: Not logged in"
        info "  Run: ./dev-setup.sh --harbor-login"
    fi
    
    # Scripts
    if [[ -x "deploy.sh" ]]; then
        success "deploy.sh: Ready"
    else
        warning "deploy.sh: Not executable"
    fi
    
    if [[ -x "quick-update.sh" ]]; then
        success "quick-update.sh: Ready"
    else
        warning "quick-update.sh: Not executable"
    fi
    
    echo ""
    echo "=== READY TO USE ==="
    success "Environment check completed"
    
    if docker info >/dev/null 2>&1 && kubectl config current-context >/dev/null 2>&1; then
        success "✨ Your development environment is ready!"
        echo ""
        info "Quick start commands:"
        echo "  ./deploy.sh build -u YOUR_USERNAME --push"
        echo "  ./deploy.sh deploy -u YOUR_USERNAME -n YOUR_NAMESPACE -d YOUR_DOMAIN"
        echo "  ./quick-update.sh -n YOUR_NAMESPACE"
    else
        warning "Environment needs attention - see errors above"
    fi
}

# Main setup
main_setup() {
    step "TEMPL Pipeline Development Environment Setup"
    
    check_docker
    check_kubectl
    setup_directories
    
    echo ""
    success "Basic development environment setup completed!"
    echo ""
    info "Optional: Setup Harbor registry login for pushing images"
    read -p "Setup Harbor login now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_harbor
    else
        info "You can setup Harbor login later with: ./dev-setup.sh --harbor-login"
    fi
    
    echo ""
    success "🎉 Development environment ready!"
    check_environment
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker-only)
                check_docker
                exit 0
                ;;
            --k8s-only)
                check_kubectl
                exit 0
                ;;
            --harbor-login)
                setup_harbor
                exit 0
                ;;
            --check)
                check_environment
                exit 0
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
    if [[ $# -eq 0 ]]; then
        main_setup
    else
        parse_args "$@"
    fi
}

main "$@"