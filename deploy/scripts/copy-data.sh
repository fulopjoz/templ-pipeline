#!/bin/bash
set -e

# Script to copy data to CERIT persistent volume

NAMESPACE=${1:-"your-namespace"}  # Replace with your actual namespace
SOURCE_DIR="./data"

echo "=========================================="
echo "Copying TEMPL Pipeline data to CERIT PVC"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Source directory: $SOURCE_DIR"
echo "=========================================="

# Check if data directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Data directory $SOURCE_DIR not found"
    exit 1
fi

# Check if PVC exists
if ! kubectl get pvc templ-data-pvc -n "$NAMESPACE" >/dev/null 2>&1; then
    echo "Error: PVC templ-data-pvc not found in namespace $NAMESPACE"
    echo "Please create the PVC first: kubectl apply -f k8s/pvc.yaml -n $NAMESPACE"
    exit 1
fi

echo "Creating temporary pod to copy data..."

# Create a temporary pod to copy data
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: data-copy-pod
  namespace: $NAMESPACE
spec:
  securityContext:
    runAsUser: 1000
    runAsNonRoot: true
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: data-copy
    image: alpine:latest
    securityContext:
      allowPrivilegeEscalation: false
      capabilities:
        drop:
        - ALL
    command: ["sleep", "3600"]
    volumeMounts:
    - mountPath: /data
      name: data-volume
  volumes:
  - name: data-volume
    persistentVolumeClaim:
      claimName: templ-data-pvc
  restartPolicy: Never
EOF

# Wait for pod to be ready
echo "Waiting for pod to be ready..."
kubectl wait --for=condition=Ready pod/data-copy-pod -n "$NAMESPACE" --timeout=60s

# Copy data files
echo "Copying data files..."
kubectl cp "$SOURCE_DIR" "$NAMESPACE/data-copy-pod:/data/" --no-preserve=true

echo "Verifying data copy..."
kubectl exec data-copy-pod -n "$NAMESPACE" -- ls -la /data/

echo "Cleaning up temporary pod..."
kubectl delete pod data-copy-pod -n "$NAMESPACE"

echo ""
echo "=========================================="
echo "Data copy completed successfully!"
echo "The data is now available in the PVC for the TEMPL Pipeline deployment."
echo "=========================================="
