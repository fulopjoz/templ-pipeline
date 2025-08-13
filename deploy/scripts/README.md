# TEMPL Pipeline Deployment Scripts

Production deployment system for TEMPL Pipeline with persistent storage.

## Scripts Overview

- **`./quick-update-persistent.sh`** - Fast code updates with persistent storage (recommended)
- **`./deploy.sh`** - Full deployment and management commands

## Development Workflow

### Daily Code Updates
```bash
# Quick code update (30 seconds)
./deploy/scripts/quick-update-persistent.sh -n fulop-ns

# Force restart if changes don't appear
./deploy/scripts/quick-update-persistent.sh -n fulop-ns --force-restart

# Preview changes before applying
./deploy/scripts/quick-update-persistent.sh -n fulop-ns --dry-run
```

### Major Updates (Docker Image Rebuild)
```bash
# Build main application image
./deploy/scripts/deploy.sh build -u xfulop --push

# Build data initialization container
./deploy/scripts/deploy.sh build-init -u xfulop --push

# Update deployment with new images
./deploy/scripts/deploy.sh update -u xfulop -n fulop-ns

# Verify deployment status
kubectl get pods -l app=templ-pipeline -n fulop-ns
```

### Data Management
```bash
# Check dataset status and storage usage
./deploy/scripts/deploy.sh data-status -n fulop-ns

# Check deployment status
./deploy/scripts/deploy.sh status -n fulop-ns

# View application logs
./deploy/scripts/deploy.sh logs -n fulop-ns

# Get shell access to running pod
./deploy/scripts/deploy.sh shell -n fulop-ns
```

## Command Reference

### Quick Code Updates
For Python code changes in `templ_pipeline/` or `scripts/` directories:

```bash
# Standard update
./deploy/scripts/quick-update-persistent.sh -n fulop-ns

# With forced restart
./deploy/scripts/quick-update-persistent.sh -n fulop-ns --force-restart

# Dry run to preview changes
./deploy/scripts/quick-update-persistent.sh -n fulop-ns --dry-run
```

### Manual Pod Management
```bash
# Force deployment restart
kubectl rollout restart deployment/templ-pipeline -n fulop-ns

# Wait for restart completion
kubectl rollout status deployment/templ-pipeline -n fulop-ns

# Check pod status
kubectl get pods -l app=templ-pipeline -n fulop-ns
```

### Monitoring and Diagnostics
```bash
# View pod status
kubectl get pods -l app=templ-pipeline -n fulop-ns

# Application logs
./deploy/scripts/deploy.sh logs -n fulop-ns

# Shell access
./deploy/scripts/deploy.sh shell -n fulop-ns

# Persistent volume status
kubectl get pvc -n fulop-ns

# Port forwarding for local access
kubectl port-forward svc/templ-pipeline-svc 8501:80 -n fulop-ns
```

## Usage Guidelines

### When to Use Quick Updates
- Python code modifications in `templ_pipeline/`
- Script updates in `scripts/`
- UI component changes
- Business logic modifications

### When to Use Full Rebuild
- New Python dependencies in `pyproject.toml`
- Docker configuration changes
- Base image updates
- System dependency modifications

### When to Use Manual Restart
- Quick update completes but changes don't appear
- Application becomes unresponsive
- Memory or resource issues

### When to Use Data Management Commands
- Check if all datasets are properly downloaded
- Monitor storage usage (approaching 50Gi limit)
- Troubleshoot missing PDBBind or Zenodo data
- Initial deployment setup verification

## Troubleshooting

### Deployment Issues
```bash
# Check pod events
kubectl describe pod -l app=templ-pipeline -n fulop-ns

# View previous container logs
kubectl logs -l app=templ-pipeline -n fulop-ns --previous

# Check resource usage
kubectl top pods -n fulop-ns
```

### Image Pull Problems
```bash
# Check image availability
docker manifest inspect cerit.io/xfulop/templ-pipeline:latest-production

# Verify Harbor login
docker login cerit.io

# Force image pull
kubectl rollout restart deployment/templ-pipeline -n fulop-ns
```

### Persistent Storage Issues
```bash
# Verify volume mounts
kubectl describe pod -l app=templ-pipeline -n fulop-ns | grep -A5 Mounts

# Check PVC status
kubectl get pvc -n fulop-ns

# Verify storage class
kubectl get storageclass
```

## Performance Notes

- Quick updates: ~30 seconds execution time
- Full rebuilds: ~20+ minutes execution time
- Code changes persist across pod restarts via persistent volumes
- Data and configuration maintained during updates