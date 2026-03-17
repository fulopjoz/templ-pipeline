#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
#
# Deploy simplified TEMPL pipeline (always-on, lean resources, no GPU).
# Usage: ./deploy.sh [--namespace fulop-ns] [--dry-run]

set -euo pipefail

NAMESPACE="${NAMESPACE:-fulop-ns}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace) NAMESPACE="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run=client"; echo "=== DRY RUN ==="; shift ;;
        -h|--help)
            echo "Usage: $0 [--namespace fulop-ns] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  TEMPL Pipeline: Simple Lean Deployment"
echo "============================================"
echo "Namespace: $NAMESPACE"
echo ""

echo ">>> 1/4: ConfigMap..."
kubectl apply -f "$SCRIPT_DIR/configmap.yaml" -n "$NAMESPACE" $DRY_RUN

echo ">>> 2/4: Service..."
kubectl apply -f "$SCRIPT_DIR/service.yaml" -n "$NAMESPACE" $DRY_RUN

echo ">>> 3/4: Deployment..."
kubectl apply -f "$SCRIPT_DIR/deployment.yaml" -n "$NAMESPACE" $DRY_RUN

echo ">>> 4/4: Ingress..."
kubectl apply -f "$SCRIPT_DIR/ingress.yaml" -n "$NAMESPACE" $DRY_RUN

# Clean up old ingress names if they exist
if [ -z "$DRY_RUN" ]; then
    kubectl delete ingress templ-pipeline-ingress-lean -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    kubectl delete ingress templ-pipeline-ingress-secure -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    kubectl delete ingress templ-pipeline-ingress-optimized -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    # Clean up scale-to-zero leftover services
    kubectl delete service templ-pipeline-app-svc -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    kubectl delete service templ-gatekeeper-svc -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
fi

echo ""
if [ -z "$DRY_RUN" ]; then
    echo "Waiting for rollout..."
    kubectl rollout status deployment/templ-pipeline -n "$NAMESPACE" --timeout=300s || true

    echo ""
    echo "--- Status ---"
    kubectl get pods -n "$NAMESPACE" -l app=templ-pipeline -o wide
    echo ""
    kubectl get ingress -n "$NAMESPACE" -o wide
    echo ""
    echo "Resources: 1 CPU request / 2 CPU limit, 4Gi / 6Gi, no GPU"
    echo "URL: https://templ.dyn.cloud.e-infra.cz"
fi
