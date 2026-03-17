# TEMPL Pipeline: Scale-to-Zero Deployment

Resource-efficient Kubernetes deployment that scales the TEMPL web app to zero when idle and starts it on-demand when users visit.

## Architecture

```
User → Ingress (nginx) → Gatekeeper Pod (always-on, ~30m CPU / 48Mi RAM)
                              ├── App is UP   → reverse proxy to TEMPL app
                              └── App is DOWN → serve "Starting up..." page
                                              → trigger kubectl scale to 1
                                              → auto-redirect when ready

CronJob (*/5 min) → checks active WebSocket connections → scales to 0 after 30min idle
```

## Resource Comparison

| State | CPU Requests | Memory | GPU |
|-------|-------------|--------|-----|
| **Original (always-on)** | 8 cores | 12Gi | 1 MIG 10GB |
| **Lean: idle** | 100m | 128Mi | none |
| **Lean: active** | ~1 core | ~4Gi | none |

## Quick Start

```bash
# Deploy (from this directory)
chmod +x deploy-lean.sh rollback.sh
./deploy-lean.sh

# Preview without applying
./deploy-lean.sh --dry-run

# Rollback to original
./rollback.sh
```

## Files

| File | Purpose |
|------|---------|
| `deployment.lean.yaml` | Main app: CPU-only, 1 core, 4Gi, replicas=0 |
| `configmap.lean.yaml` | App config without GPU variables |
| `service.lean.yaml` | ClusterIP service for main app |
| `gatekeeper-deployment.yaml` | Always-on nginx proxy + kubectl waker sidecar |
| `gatekeeper-configmap.yaml` | nginx.conf + branded loading page HTML |
| `gatekeeper-service.yaml` | ClusterIP service for gatekeeper |
| `ingress.lean.yaml` | Ingress pointing to gatekeeper (same domain/TLS) |
| `idle-scaler-rbac.yaml` | ServiceAccount + Role + RoleBinding |
| `idle-scaler-cronjob.yaml` | CronJob: scales to 0 after 30min idle |
| `deploy-lean.sh` | Deployment script |
| `rollback.sh` | Rollback to original deployment |

## How It Works

### Wake-up Flow
1. User visits `https://templ.dyn.cloud.e-infra.cz`
2. Ingress routes to gatekeeper (always running)
3. Gatekeeper tries to proxy to main app → gets 502 (app is at 0 replicas)
4. Gatekeeper serves the loading page HTML instead
5. Loading page JavaScript calls `/api/wake`
6. Gatekeeper's waker sidecar receives the request, runs `kubectl scale --replicas=1`
7. Loading page polls `/api/status` every 5 seconds
8. When app is ready (~60-90s), page auto-redirects to the app

### Idle Scale-down Flow
1. CronJob runs every 5 minutes
2. Checks established TCP connections on port 8501 (Streamlit WebSocket connections)
3. If active connections > 2: updates `templ/last-active` annotation on deployment
4. If connections <= 2 and last-active > 30 minutes ago: scales to 0

### Why connections > 2?
Kubernetes health probes (readiness + liveness) create short-lived HTTP connections. A threshold of >2 filters these out while detecting real user sessions (Streamlit maintains persistent WebSocket connections per browser tab).

## Configuration

| Parameter | Default | Where |
|-----------|---------|-------|
| Idle timeout | 30 min (1800s) | `idle-scaler-cronjob.yaml` → `IDLE_TIMEOUT` |
| Connection threshold | 2 | `idle-scaler-cronjob.yaml` → comparison |
| Check interval | 5 min | `idle-scaler-cronjob.yaml` → `schedule` |
| Poll interval (loading page) | 5s | `gatekeeper-configmap.yaml` → JS |
| App CPU request/limit | 1/2 | `deployment.lean.yaml` |
| App memory request/limit | 4Gi/6Gi | `deployment.lean.yaml` |

## Troubleshooting

```bash
# Check gatekeeper status
kubectl get pods -l app=templ-gatekeeper -n fulop-ns
kubectl logs -l app=templ-gatekeeper -c nginx -n fulop-ns
kubectl logs -l app=templ-gatekeeper -c waker -n fulop-ns

# Check main app status
kubectl get deployment templ-pipeline -n fulop-ns
kubectl get pods -l app=templ-pipeline -n fulop-ns

# Check idle scaler
kubectl get cronjob templ-idle-scaler -n fulop-ns
kubectl get jobs -l app=templ-pipeline -n fulop-ns --sort-by=.metadata.creationTimestamp

# Manually scale up/down
kubectl scale deployment/templ-pipeline --replicas=1 -n fulop-ns
kubectl scale deployment/templ-pipeline --replicas=0 -n fulop-ns

# Check last-active annotation
kubectl get deployment templ-pipeline -n fulop-ns -o jsonpath='{.metadata.annotations.templ/last-active}'
```

## Prerequisites

- CERIT-SC Kubernetes cluster access with `fulop-ns` namespace
- Permission to create Roles and RoleBindings in namespace
- Images pullable: `nginx:1.27-alpine`, `bitnami/kubectl:1.31`
- Existing resources: `templ-pipeline-secrets` Secret, TLS cert for domain
