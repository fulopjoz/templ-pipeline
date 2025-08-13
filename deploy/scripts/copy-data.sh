#!/bin/bash
set -e

# Enhanced script to copy TEMPL Pipeline data to CERIT persistent volume
# Supports large datasets including PDBBind data (~12GB)

NAMESPACE=${1:-"your-namespace"}  # Replace with your actual namespace
SOURCE_DIR="./data"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO] $1${NC}"; }
success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
warning() { echo -e "${YELLOW}[WARNING] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; }

echo "=========================================="
echo "Enhanced TEMPL Pipeline Data Copy to PVC"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Source directory: $SOURCE_DIR"
echo "Target PVC: templ-data-pvc (50Gi)"
echo "=========================================="

# Check if data directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    error "Data directory $SOURCE_DIR not found"
    info "Expected structure:"
    echo "  data/"
    echo "  ├── embeddings/"
    echo "  ├── ligands/"
    echo "  └── PDBBind/"
    exit 1
fi

# Show data directory size
if command -v du >/dev/null 2>&1; then
    info "Source data size:"
    du -sh "$SOURCE_DIR"/* 2>/dev/null | sort -hr || true
    echo ""
fi

# Check if PVC exists and show status
if ! kubectl get pvc templ-data-pvc -n "$NAMESPACE" >/dev/null 2>&1; then
    error "PVC templ-data-pvc not found in namespace $NAMESPACE"
    info "Create the PVC first: kubectl apply -f deploy/kubernetes/pvc.yaml -n $NAMESPACE"
    exit 1
fi

# Show PVC status
info "PVC Status:"
kubectl get pvc templ-data-pvc -n "$NAMESPACE"
echo ""

info "Creating temporary pod for data transfer..."

# Create a temporary pod with better resources for large data transfers
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: data-copy-pod
  namespace: $NAMESPACE
  labels:
    app: templ-data-copy
spec:
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    runAsNonRoot: true
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: data-copy
    image: alpine:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: false
      capabilities:
        drop:
        - ALL
    command: ["sleep", "7200"]  # 2 hours for large transfers
    resources:
      requests:
        memory: "256Mi"
        cpu: "250m"
      limits:
        memory: "512Mi"
        cpu: "500m"
    volumeMounts:
    - mountPath: /data
      name: data-volume
    - mountPath: /tmp
      name: temp-volume
  volumes:
  - name: data-volume
    persistentVolumeClaim:
      claimName: templ-data-pvc
  - name: temp-volume
    emptyDir:
      sizeLimit: 1Gi
  restartPolicy: Never
EOF

# Wait for pod to be ready
info "Waiting for pod to be ready..."
kubectl wait --for=condition=Ready pod/data-copy-pod -n "$NAMESPACE" --timeout=120s

# Copy data files with progress tracking
info "Copying data files to persistent volume..."
info "This may take several minutes for large datasets..."

# Copy each subdirectory separately for better progress tracking
for subdir in "$SOURCE_DIR"/*; do
    if [[ -d "$subdir" ]]; then
        subdir_name=$(basename "$subdir")
        info "Copying $subdir_name/..."
        
        # Get size for progress indication
        if command -v du >/dev/null 2>&1; then
            size=$(du -sh "$subdir" | cut -f1)
            info "  Size: $size"
        fi
        
        kubectl cp "$subdir" "$NAMESPACE/data-copy-pod:/data/" --no-preserve=true
        success "  $subdir_name/ copied successfully"
    fi
done

info "Verifying data copy..."
kubectl exec data-copy-pod -n "$NAMESPACE" -- sh -c "
    echo 'Data directory contents:'
    ls -la /data/
    echo ''
    echo 'Disk usage summary:'
    du -sh /data/* 2>/dev/null | sort -hr || true
"

info "Cleaning up temporary pod..."
kubectl delete pod data-copy-pod -n "$NAMESPACE"

success "=========================================="
success "Enhanced data copy completed successfully!"
success "=========================================="
info "The following data is now available in the PVC:"
info "- Zenodo datasets (embeddings, ligands)"
info "- PDBBind datasets (~12GB when extracted)"
info "- Total storage: up to 50Gi available"
success "=========================================="
