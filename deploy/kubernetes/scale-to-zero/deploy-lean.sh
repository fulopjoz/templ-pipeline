#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
#
# Deploy the TEMPL pipeline scale-to-zero architecture.
# Usage: ./deploy-lean.sh [--namespace fulop-ns] [--dry-run]
#
# This script:
#   1. Applies all scale-to-zero manifests in the correct order
#   2. Removes old ingress resources (replaced by gatekeeper-based ingress)
#   3. Verifies the gatekeeper is running
#   4. Prints a summary of the new deployment state

set -euo pipefail

NAMESPACE="${NAMESPACE:-fulop-ns}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run=client"
            echo "=== DRY RUN MODE ==="
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--namespace fulop-ns] [--dry-run]"
            echo ""
            echo "Deploys the TEMPL pipeline scale-to-zero architecture."
            echo "All existing manifests are preserved for rollback."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  TEMPL Pipeline: Scale-to-Zero Deployment"
echo "============================================"
echo "Namespace: $NAMESPACE"
echo "Manifests: $SCRIPT_DIR"
echo ""

# Step 1: RBAC (must be first — other resources depend on the ServiceAccount)
echo ">>> Step 1/7: Applying RBAC..."
kubectl apply -f "$SCRIPT_DIR/idle-scaler-rbac.yaml" -n "$NAMESPACE" $DRY_RUN

# Step 2: ConfigMap (before deployment so env vars are available)
echo ">>> Step 2/7: Applying ConfigMaps..."
kubectl apply -f "$SCRIPT_DIR/configmap.lean.yaml" -n "$NAMESPACE" $DRY_RUN
kubectl apply -f "$SCRIPT_DIR/gatekeeper-configmap.yaml" -n "$NAMESPACE" $DRY_RUN

# Step 3: Services (before deployments so DNS is ready)
echo ">>> Step 3/7: Applying Services..."
kubectl apply -f "$SCRIPT_DIR/service.lean.yaml" -n "$NAMESPACE" $DRY_RUN
kubectl apply -f "$SCRIPT_DIR/gatekeeper-service.yaml" -n "$NAMESPACE" $DRY_RUN

# Step 4: Main app deployment (starts at replicas=0)
echo ">>> Step 4/7: Applying Main App Deployment (replicas=0)..."
kubectl apply -f "$SCRIPT_DIR/deployment.lean.yaml" -n "$NAMESPACE" $DRY_RUN

# Step 5: Gatekeeper deployment (always-on proxy)
echo ">>> Step 5/7: Applying Gatekeeper Deployment..."
kubectl apply -f "$SCRIPT_DIR/gatekeeper-deployment.yaml" -n "$NAMESPACE" $DRY_RUN

# Step 6: Ingress (point to gatekeeper, then remove old ingress)
echo ">>> Step 6/7: Applying Ingress..."
kubectl apply -f "$SCRIPT_DIR/ingress.lean.yaml" -n "$NAMESPACE" $DRY_RUN
if [ -z "$DRY_RUN" ]; then
    echo "    Removing old ingress resources..."
    kubectl delete ingress templ-pipeline-ingress -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    kubectl delete ingress templ-pipeline-ingress-secure -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    kubectl delete ingress templ-pipeline-ingress-optimized -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
fi

# Step 7: CronJob (idle scaler)
echo ">>> Step 7/7: Applying Idle Scaler CronJob..."
kubectl apply -f "$SCRIPT_DIR/idle-scaler-cronjob.yaml" -n "$NAMESPACE" $DRY_RUN

echo ""
echo "============================================"
echo "  Deployment Complete"
echo "============================================"

if [ -z "$DRY_RUN" ]; then
    echo ""
    echo "Waiting for gatekeeper pod to be ready..."
    kubectl rollout status deployment/templ-gatekeeper -n "$NAMESPACE" --timeout=120s || true

    echo ""
    echo "--- Current State ---"
    echo ""
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -l 'app in (templ-pipeline, templ-gatekeeper)' -o wide
    echo ""
    echo "Services:"
    kubectl get svc -n "$NAMESPACE" -l 'app in (templ-pipeline, templ-gatekeeper)' -o wide
    echo ""
    echo "Ingress:"
    kubectl get ingress -n "$NAMESPACE" -o wide
    echo ""
    echo "CronJobs:"
    kubectl get cronjob -n "$NAMESPACE" -o wide
    echo ""
    echo "--- Resource Summary ---"
    echo "Gatekeeper: ~100m CPU, ~128Mi RAM (always on, 2 containers)"
    echo "Main app:   0 replicas (scales to 1 on user visit)"
    echo "            1 CPU / 4Gi RAM when running (no GPU)"
    echo "Idle timeout: 30 minutes"
    echo ""
    echo "Visit https://templ.dyn.cloud.e-infra.cz to test."
    echo "To rollback: ./rollback.sh"
fi
