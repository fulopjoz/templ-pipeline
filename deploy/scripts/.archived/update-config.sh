#!/bin/bash
set -e

# TEMPL Pipeline Configuration Update Script
# Updates ConfigMap and restarts deployment without rebuilding Docker image
# This provides near-instant configuration updates (30 seconds vs 20+ minutes)

echo "=========================================="
echo "TEMPL Pipeline Configuration Update"
echo "Updating ConfigMap without Docker rebuild"
echo "=========================================="

# Apply the ConfigMap
echo "Applying ConfigMap..."
kubectl apply -f deploy/kubernetes/configmap.yaml

# Verify ConfigMap was created/updated
echo "Verifying ConfigMap..."
kubectl get configmap templ-pipeline-config -o yaml

# Update deployment to use ConfigMap (if not already configured)
echo "Applying deployment configuration..."
kubectl apply -f deploy/kubernetes/deployment.yaml

# Restart the deployment to pick up new configuration
echo "Restarting deployment to pick up new configuration..."
kubectl rollout restart deployment templ-pipeline

# Wait for rollout to complete
echo "Waiting for rollout to complete..."
kubectl rollout status deployment templ-pipeline --timeout=120s

# Verify deployment
echo "Verifying deployment..."
kubectl get pods -l app=templ-pipeline

# Show the environment variables in the running pod
echo ""
echo "Verifying environment variables in running pod..."
POD_NAME=$(kubectl get pods -l app=templ-pipeline -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD_NAME"
echo ""
echo "Key environment variables:"
kubectl exec $POD_NAME -- env | grep -E "TEMPL_(EMBEDDING|LIGANDS|DATA)_PATH|STREAMLIT_" | sort

echo ""
echo "=========================================="
echo "✅ Configuration update completed!"
echo "=========================================="
echo ""
echo "Benefits of this approach:"
echo "• ⚡ 30 seconds vs 20+ minutes for Docker rebuild"  
echo "• 🔧 Change paths, ports, and settings instantly"
echo "• 🔄 No image rebuild required for config changes"
echo "• 📝 Version controlled configuration"
echo ""
echo "To change configuration:"
echo "1. Edit deploy/kubernetes/configmap.yaml"
echo "2. Run: ./deploy/scripts/update-config.sh"
echo "3. Changes applied in 30 seconds!"