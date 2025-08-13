# TEMPL Pipeline Data Management Guide

## Overview

The TEMPL Pipeline requires several large datasets for optimal functionality:

- **Protein Embeddings**: ~2GB (Zenodo)
- **Processed Ligands**: ~500MB (Zenodo)  
- **PDBBind Dataset**: ~12GB (CESNET mirror)

Total storage requirement: ~15GB + working space (50Gi PVC recommended)

## Architecture

### Storage Strategy
- **Persistent Volumes**: 50Gi NFS-based storage for dataset persistence
- **Init Containers**: Automated data download and preparation
- **Volume Sharing**: Data shared between init and main application containers

### Data Sources
1. **Zenodo Repository**: `10.5281/zenodo.15813500`
   - Protein embeddings (`templ_protein_embeddings_v1.0.0.npz`)
   - Processed ligands (`templ_processed_ligands_v1.0.0.sdf.gz`)

2. **CESNET Mirror**: PDBBind v2020 datasets
   - Refined set: `REMOVED_PDBBIND_REFINED_URL`
   - Other PL set: `REMOVED_PDBBIND_OTHER_PL_URL`

## Deployment Workflow

### Initial Setup (First Time)

1. **Expand PVC Storage**
   ```bash
   # Update PVC to 50Gi
   kubectl apply -f deploy/kubernetes/pvc.yaml -n fulop-ns
   ```

2. **Build Init Container**
   ```bash
   # Build data initialization container
   ./deploy/scripts/build-init-container.sh
   ```

3. **Deploy with Data Initialization**
   ```bash
   # Deploy with init container (downloads data automatically)
   kubectl apply -f deploy/kubernetes/deployment.persistent.yaml -n fulop-ns
   ```

4. **Monitor Data Download**
   ```bash
   # Watch init container progress
   kubectl logs -f -l app=templ-pipeline -c data-init -n fulop-ns
   
   # Check main container startup
   kubectl logs -f -l app=templ-pipeline -c templ-pipeline -n fulop-ns
   ```

### Data Management Commands

#### Check Data Status
```bash
# Verify data presence in PVC
kubectl exec deployment/templ-pipeline -n fulop-ns -- ls -la /app/data/

# Check storage usage
kubectl exec deployment/templ-pipeline -n fulop-ns -- du -sh /app/data/*
```

#### Manual Data Copy (Alternative)
```bash
# Copy local data to PVC (if you have datasets locally)
./deploy/scripts/copy-data.sh fulop-ns

# Verify copy completion
kubectl get pods -l app=templ-data-copy -n fulop-ns
```

#### Force Data Re-download
```bash
# Delete deployment to trigger init container restart
kubectl delete deployment templ-pipeline -n fulop-ns

# Redeploy (init container will check and download missing data)
kubectl apply -f deploy/kubernetes/deployment.persistent.yaml -n fulop-ns
```

## File Structure

Expected data directory structure in persistent volume:

```
/app/data/
├── embeddings/
│   └── templ_protein_embeddings_v1.0.0.npz
├── ligands/
│   └── templ_processed_ligands_v1.0.0.sdf.gz
└── PDBBind/
    ├── PDBbind_v2020_refined/
    └── PDBbind_v2020_other_PL/
```

## Troubleshooting

### Init Container Issues

**Problem**: Init container fails to download data
```bash
# Check init container logs
kubectl logs -l app=templ-pipeline -c data-init -n fulop-ns

# Check network connectivity
kubectl exec deployment/templ-pipeline -n fulop-ns -- ping zenodo.org

# Manually trigger data download
kubectl exec deployment/templ-pipeline -n fulop-ns -- python3 /app/scripts/setup_pdbind_data.py
```

**Problem**: Storage full during download
```bash
# Check PVC usage
kubectl exec deployment/templ-pipeline -n fulop-ns -- df -h /app/data

# Expand PVC if needed (update pvc.yaml and reapply)
kubectl apply -f deploy/kubernetes/pvc.yaml -n fulop-ns
```

### Data Integrity Issues

**Problem**: Application reports missing datasets
```bash
# Verify all required files exist
kubectl exec deployment/templ-pipeline -n fulop-ns -- find /app/data -name "*.npz" -o -name "*.sdf.gz"

# Check PDBBind structure
kubectl exec deployment/templ-pipeline -n fulop-ns -- ls -la /app/data/PDBBind/
```

**Problem**: Corrupted downloads
```bash
# Clear data and re-download
kubectl exec deployment/templ-pipeline -n fulop-ns -- rm -rf /app/data/*

# Restart deployment to trigger re-download
kubectl rollout restart deployment/templ-pipeline -n fulop-ns
```

## Performance Considerations

### Download Times
- **Zenodo datasets**: ~5-10 minutes (good connectivity)
- **PDBBind datasets**: ~30-45 minutes (12GB compressed)
- **Total setup time**: ~45-60 minutes (first deployment)

### Storage Optimization
- **Compression**: PDBBind data is downloaded compressed and extracted in-place
- **Cleanup**: Temporary download files are automatically removed
- **Caching**: Data persists across pod restarts and deployments

### Resource Requirements
- **Init Container**: 1Gi memory, 1 CPU core recommended
- **Network**: Stable internet connection required for initial setup
- **Storage**: 50Gi PVC provides ~35Gi working space after datasets

## Integration with CI/CD

### Automated Deployment
```bash
# Build both main and init containers
./deploy/scripts/deploy.sh build -u xfulop --push
./deploy/scripts/build-init-container.sh

# Deploy with automatic data setup
./deploy/scripts/deploy.sh deploy -u xfulop -n fulop-ns -d your-domain
```

### Data Validation
The init container automatically validates all downloads and reports missing or corrupted files. Check init container logs for validation results.

### Backup and Recovery
```bash
# Backup data from PVC
./deploy/scripts/copy-data.sh fulop-ns  # Reverse direction for backup

# Restore data to new PVC
kubectl apply -f deploy/kubernetes/pvc.yaml -n new-namespace
./deploy/scripts/copy-data.sh new-namespace
```