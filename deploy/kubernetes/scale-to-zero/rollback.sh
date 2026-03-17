#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
#
# Rollback from scale-to-zero to the original optimized deployment.
# Usage: ./rollback.sh [--namespace fulop-ns]
#
# This script:
#   1. Deletes all scale-to-zero resources (gatekeeper, idle scaler, RBAC)
#   2. Re-applies the original manifests from deploy/kubernetes/
#   3. Scales the main deployment back to 1 replica

set -euo pipefail

NAMESPACE="${NAMESPACE:-fulop-ns}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_DIR="$(dirname "$SCRIPT_DIR")"  # deploy/kubernetes/

echo "============================================"
echo "  TEMPL Pipeline: Rollback to Original"
echo "============================================"
echo "Namespace: $NAMESPACE"
echo ""

read -p "This will remove all scale-to-zero resources and restore the original deployment. Continue? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Rollback cancelled."
    exit 0
fi

echo ""

# Step 1: Delete scale-to-zero resources
echo ">>> Step 1/4: Removing scale-to-zero resources..."
kubectl delete cronjob templ-idle-scaler -n "$NAMESPACE" --ignore-not-found
kubectl delete ingress templ-pipeline-ingress-lean -n "$NAMESPACE" --ignore-not-found
kubectl delete deployment templ-gatekeeper -n "$NAMESPACE" --ignore-not-found
kubectl delete service templ-gatekeeper-svc -n "$NAMESPACE" --ignore-not-found
kubectl delete service templ-pipeline-app-svc -n "$NAMESPACE" --ignore-not-found
kubectl delete configmap templ-gatekeeper-config -n "$NAMESPACE" --ignore-not-found
kubectl delete rolebinding templ-scaler-binding -n "$NAMESPACE" --ignore-not-found
kubectl delete role templ-scaler-role -n "$NAMESPACE" --ignore-not-found
kubectl delete serviceaccount templ-scaler -n "$NAMESPACE" --ignore-not-found

# Step 2: Re-apply original manifests
echo ">>> Step 2/4: Applying original ConfigMap..."
kubectl apply -f "$ORIGINAL_DIR/configmap.optimized.yaml" -n "$NAMESPACE"

echo ">>> Step 3/4: Applying original Service and Ingress..."
kubectl apply -f "$ORIGINAL_DIR/service.optimized.yaml" -n "$NAMESPACE"
kubectl apply -f "$ORIGINAL_DIR/ingress-secure.yaml" -n "$NAMESPACE"

echo ">>> Step 4/4: Restoring original Deployment..."
kubectl apply -f "$ORIGINAL_DIR/deployment.optimized.yaml" -n "$NAMESPACE"

echo ""
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/templ-pipeline -n "$NAMESPACE" --timeout=300s || true

echo ""
echo "============================================"
echo "  Rollback Complete"
echo "============================================"
echo ""
echo "Pods:"
kubectl get pods -n "$NAMESPACE" -l app=templ-pipeline -o wide
echo ""
echo "Ingress:"
kubectl get ingress -n "$NAMESPACE" -o wide
echo ""
echo "WARNING: The original deployment uses 8 CPU + 12Gi RAM + 1 MIG GPU."
echo "Consider reducing resources if the waste alert was the reason for this rollback."
