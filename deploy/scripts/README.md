# TEMPL Pipeline Deployment Scripts

Production deployment system for TEMPL Pipeline with persistent storage.

## Scripts Overview

- **`./quick-update-persistent.sh`** - Fast code updates with persistent storage (recommended)
- **`./deploy.sh`** - Full deployment and management commands

## Quick Development Workflow

### Code Updates (Most Common)
```bash
# 1. Edit your Python code
vim templ_pipeline/ui/components/input_section.py

# 2. Quick update (30 seconds)
./quick-update-persistent.sh -n fulop-ns

# 3. If changes don't appear, force restart
kubectl rollout restart deployment/templ-pipeline -n fulop-ns

# 4. Check status
kubectl get pods -l app=templ-pipeline -n fulop-ns
```

## Essential Commands

### Quick Code Updates
```bash
# Update code with persistent storage
./quick-update-persistent.sh -n fulop-ns

# Update with forced restart (ensures changes apply)
./quick-update-persistent.sh -n fulop-ns --force-restart

# Preview what would be updated
./quick-update-persistent.sh -n fulop-ns --dry-run
```

### Manual Pod Restart (when changes don't appear)
```bash
# Force deployment restart
kubectl rollout restart deployment/templ-pipeline -n fulop-ns

# Wait for restart to complete
kubectl rollout status deployment/templ-pipeline -n fulop-ns

# Check pod status
kubectl get pods -l app=templ-pipeline -n fulop-ns
```

### Monitoring and Troubleshooting
```bash
# Check pod status
kubectl get pods -l app=templ-pipeline -n fulop-ns

# View application logs
kubectl logs -l app=templ-pipeline -n fulop-ns --tail=100

# Get shell access
kubectl exec -it deployment/templ-pipeline -n fulop-ns -- /bin/bash

# Check persistent volumes
kubectl get pvc -n fulop-ns

# Port forward for local access
kubectl port-forward svc/templ-pipeline-svc 8501:80 -n fulop-ns
```

### Full Deployment (First Time or Major Changes)
```bash
# Complete deployment
./deploy.sh deploy -u xfulop -n fulop-ns -d your-domain

# Check deployment status
./deploy.sh status -n fulop-ns

# View logs
./deploy.sh logs -n fulop-ns
```

## When to Use What

**Use `./quick-update-persistent.sh` for:**
- Python code changes in `templ_pipeline/`
- Script updates in `scripts/`
- UI modifications
- Business logic changes

**Use `kubectl rollout restart` when:**
- Quick update completes but changes don't appear in the app
- Need to force application restart

**Use full deployment for:**
- New Python dependencies in `pyproject.toml`
- Dockerfile changes
- Kubernetes configuration changes

## Troubleshooting

### Changes Not Showing After Quick Update
```bash
# Force pod restart
kubectl rollout restart deployment/templ-pipeline -n fulop-ns
kubectl rollout status deployment/templ-pipeline -n fulop-ns
```

### Pod Won't Start
```bash
# Check pod events
kubectl describe pod -l app=templ-pipeline -n fulop-ns

# Check logs for errors
kubectl logs -l app=templ-pipeline -n fulop-ns --previous
```

### Check Data Persistence
```bash
# Verify persistent volumes are mounted
kubectl describe pod -l app=templ-pipeline -n fulop-ns | grep -A5 Mounts

# Check PVC status
kubectl get pvc -n fulop-ns
```

---

**Performance:** Quick updates take ~30 seconds vs 20+ minutes for full rebuild.
**Data:** All data persists across pod restarts using persistent volumes.