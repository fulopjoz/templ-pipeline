# TEMPL Pipeline Deployment Infrastructure

This directory contains Docker and Kubernetes configurations for deploying the TEMPL web application.

---

## For Users

**Recommended access methods:**

1. **Hosted Web Application**: [templ.dyn.cloud.e-infra.cz](https://templ.dyn.cloud.e-infra.cz/) (no setup required)
2. **Command-Line Interface**: Install locally and use `templ run` commands for pose prediction
3. **Local Web UI**: Run `python scripts/run_streamlit_app.py` (single command, local access)

See the main [README.md](../README.md) for installation and usage instructions.

---

## For System Administrators

This directory provides reference implementations for deploying TEMPL in production environments:

- **`docker/`** - Docker container configuration with multi-stage builds
- **`kubernetes/`** - Kubernetes deployment manifests (e-INFRA CZ platform specific)
- **`scripts/`** - Operational scripts for deployment management and updates

### Current Production Deployment

The production instance at [templ.dyn.cloud.e-infra.cz](https://templ.dyn.cloud.e-infra.cz/) runs on the [e-INFRA CZ](https://www.e-infra.cz/) platform using these configurations.

**Infrastructure details:**
- **Platform**: CERIT-SC Kubernetes cluster
- **Registry**: Harbor registry at `hub.cerit.io`
- **Storage**: Persistent volumes with 50Gi capacity
- **Resources**: 2-4Gi memory, 0.5-2 CPU cores per pod
- **Data**: Zenodo datasets embedded, PDBbind mounted externally

### Data Handling

The deployment architecture respects data licensing requirements:

- **Embedded in Docker images**: Freely distributable datasets from Zenodo (DOI: 10.5281/zenodo.15813500)
  - Protein embeddings (`templ_protein_embeddings_v1.0.0.npz`, ~90MB)
  - Processed ligands (`templ_processed_ligands_v1.0.0.sdf.gz`, ~10MB)

- **External persistent storage**: PDBbind dataset (NOT included in images)
  - Must be manually configured per licensing terms
  - See main `README.md` for download instructions from [pdbbind-plus.org.cn](https://www.pdbbind-plus.org.cn/download)
  - Mounted as read-only persistent volume in Kubernetes

This approach ensures compliance with PDBbind licensing while maintaining reproducible containerized deployments.

---

## Adapting for Your Institution

These configurations are provided as reference implementations. Key considerations for adaptation:

### 1. Container Registry
Update image references in Kubernetes manifests:
```yaml
# Current (e-INFRA specific):
image: hub.cerit.io/xfulop/templ-pipeline:latest-production

# Change to your registry:
image: your-registry.example.com/your-org/templ-pipeline:version
```

### 2. Persistent Storage
Configure storage appropriate for your infrastructure:
```yaml
# Adjust storage class and capacity in code-pvc.yaml:
storageClassName: your-storage-class  # e.g., nfs-client, local-path
resources:
  requests:
    storage: 50Gi  # Adjust based on your PDBbind subset needs
```

### 3. Authentication & Secrets
Generate institution-specific secrets:
```bash
# Password hash for web app authentication
python -c "import bcrypt; print(bcrypt.hashpw(b'your-password', bcrypt.gensalt()).decode())"

# Streamlit cookie secret (random 64-char string)
python -c "import secrets; print(secrets.token_hex(32))"
```

Update `kubernetes/secrets.template.yaml` with your values.

### 4. Resource Limits
Adjust CPU/memory based on expected workload:
```yaml
resources:
  requests:
    memory: "2Gi"    # Minimum for startup
    cpu: "500m"
  limits:
    memory: "4Gi"    # Adjust based on concurrent users
    cpu: "2"         # Adjust based on available CPU
```

### 5. Data Volume Configuration
Ensure PDBbind data is accessible:
- **Option A**: Pre-populate persistent volume with PDBbind dataset
- **Option B**: Use init container to download/copy data on first deployment
- **Option C**: Mount existing NFS/shared storage with PDBbind data

See `scripts/init-data-setup.sh` for data initialization reference.

---

## Operational Documentation

For detailed operational procedures specific to the e-INFRA CZ deployment:

- **Deployment management**: See [`scripts/README.md`](scripts/README.md)
  - Quick code updates
  - Image rebuilds
  - Data management
  - Monitoring and troubleshooting

---

## Architecture Overview

### Docker Multi-Stage Build
The Dockerfile uses a two-stage build process:

1. **Builder stage**: Installs scientific computing dependencies (RDKit, PyTorch with CUDA)
2. **Production stage**: Copies only runtime environment and application code

This approach minimizes final image size while maintaining CUDA-enabled PyTorch for GPU acceleration.

### Kubernetes Deployment Pattern
- **Single replica**: Streamlit app with persistent storage
- **Security**: Non-root user (UID 1000), dropped capabilities, security context
- **Health checks**: Liveness and readiness probes using Streamlit's `/_stcore/health` endpoint
- **Persistent volumes**: Data volume (read-only) + ephemeral temp volume

### Data Initialization
For persistent deployments, data is initialized once:
- **Init container** (optional): Runs `scripts/init-data-setup.sh` to download Zenodo datasets
- **Persistent volume**: Stores downloaded data across pod restarts
- **Application container**: Mounts pre-populated data volume

