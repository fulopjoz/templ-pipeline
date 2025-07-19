# TEMPL Pipeline CERIT Deployment Guide

This guide provides step-by-step instructions for deploying the TEMPL Pipeline on the CERIT-SC Kubernetes platform.

## Prerequisites

1. **CERIT Access**: You need access to the CERIT-SC platform with an academic affiliation
2. **Harbor Account**: Access to Harbor registry at `hub.cerit.io`
3. **kubectl**: Configured to access the CERIT cluster
4. **Docker**: For building the container image
5. **Data Files**: The complete TEMPL data directory (~3GB)

## Quick Start

### 1. Login to Harbor Registry

```bash
# Get your CLI secret from Harbor profile at hub.cerit.io
docker login hub.cerit.io
# Enter your Harbor username and CLI secret
```

### 2. Build and Push Docker Image

```bash
# Build the image (replace 'your-username' with your Harbor username)
./scripts/build.sh latest your-username

# Push to Harbor
docker push hub.cerit.io/your-username/templ-pipeline:latest
```

### 3. Prepare Data

```bash
# Copy data to CERIT persistent volume (replace 'your-namespace' with your actual namespace)
./scripts/copy-data.sh your-namespace
```

### 4. Deploy to CERIT

```bash
# Deploy the application (replace 'your-namespace' with your actual namespace)
./scripts/deploy.sh your-namespace your-username latest
```

### 5. Access Your Application

Your application will be available at: `https://templ-pipeline.dyn.cloud.e-infra.cz`

**Important**: Change the domain name in `k8s/ingress.yaml` to a unique name before deployment.

## Detailed Deployment Steps

### Step 1: Customize Configuration

1. **Update Ingress Domain**: Edit `k8s/ingress.yaml` and change `templ-pipeline.dyn.cloud.e-infra.cz` to your desired unique domain name ending with `.dyn.cloud.e-infra.cz`.

2. **Update Image Reference**: Edit `k8s/deployment.yaml` and replace `[your-username]` with your actual Harbor username.

3. **Adjust Resources**: Modify resource limits in `k8s/deployment.yaml` if needed:
   ```yaml
   resources:
     requests:
       memory: "2Gi"
       cpu: "500m"
     limits:
       memory: "4Gi"
       cpu: "2"
   ```

### Step 2: Build Docker Image

```bash
# Navigate to project directory
cd /path/to/templ_pipeline

# Build image with version tag
./scripts/build.sh v1.0.0 your-harbor-username

# Verify image was built
docker images | grep templ-pipeline
```

### Step 3: Push to Harbor

```bash
# Login to Harbor
docker login hub.cerit.io

# Push the image
docker push hub.cerit.io/your-harbor-username/templ-pipeline:v1.0.0
docker push hub.cerit.io/your-harbor-username/templ-pipeline:latest
```

### Step 4: Create Kubernetes Resources

```bash
# Apply PVC first
kubectl apply -f k8s/pvc.yaml -n your-namespace

# Wait for PVC to be bound
kubectl wait --for=condition=Bound pvc/templ-data-pvc -n your-namespace --timeout=60s

# Check PVC status
kubectl get pvc -n your-namespace
```

### Step 5: Copy Data to Persistent Volume

```bash
# Run the data copy script
./scripts/copy-data.sh your-namespace

# Verify data was copied
kubectl run --rm -it --restart=Never data-check --image=alpine:latest \
  --overrides='{"spec":{"securityContext":{"runAsUser":1000,"runAsNonRoot":true},"containers":[{"name":"data-check","image":"alpine:latest","command":["sh"],"securityContext":{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]}},"volumeMounts":[{"mountPath":"/data","name":"data-volume"}]}],"volumes":[{"name":"data-volume","persistentVolumeClaim":{"claimName":"templ-data-pvc"}}]}}' \
  -n your-namespace -- ls -la /data/
```

### Step 6: Deploy Application

```bash
# Deploy all resources
kubectl apply -f k8s/ -n your-namespace

# Check deployment status
kubectl get pods,svc,ingress -l app=templ-pipeline -n your-namespace

# Watch deployment progress
kubectl rollout status deployment/templ-pipeline -n your-namespace
```

### Step 7: Verify Deployment

```bash
# Check pod logs
kubectl logs -f deployment/templ-pipeline -n your-namespace

# Check pod details
kubectl describe pod -l app=templ-pipeline -n your-namespace

# Test health endpoint
kubectl port-forward svc/templ-pipeline-svc 8080:80 -n your-namespace
# Then visit http://localhost:8080/_stcore/health
```

## File Structure

```
templ_pipeline/
├── Dockerfile                      # Multi-stage Docker build
├── docker-entrypoint.sh           # Container entry point
├── docker-compose.yml             # Local development
├── docker-compose.prod.yml        # Production compose
├── .dockerignore                  # Docker build exclusions
├── k8s/                           # Kubernetes manifests
│   ├── deployment.yaml            # Main application deployment
│   ├── service.yaml               # Service definition
│   ├── ingress.yaml               # Ingress for external access
│   └── pvc.yaml                   # Persistent volume claim
├── scripts/                       # Deployment scripts
│   ├── build.sh                   # Docker build script
│   ├── deploy.sh                  # Kubernetes deployment
│   └── copy-data.sh              # Data copy utility
└── data/                          # Application data (~3GB)
    ├── embeddings/
    ├── ligands/
    └── ...
```

## Resource Requirements

### Minimum Requirements
- **CPU**: 500m (0.5 cores)
- **Memory**: 2Gi
- **Storage**: 5Gi for data + 2Gi for temp files

### Recommended for Production
- **CPU**: 2 cores
- **Memory**: 4Gi
- **Storage**: 10Gi for data + 5Gi for temp files

## Environment Variables

The deployment uses these key environment variables:

```yaml
TEMPL_ENV: "production"
STREAMLIT_SERVER_HEADLESS: "true"
STREAMLIT_SERVER_PORT: "8501"
STREAMLIT_SERVER_ADDRESS: "0.0.0.0"
STREAMLIT_GLOBAL_LOG_LEVEL: "info"
TEMPL_DATA_DIR: "/app/data"
TEMPL_TEMP_DIR: "/app/temp"
```

## Troubleshooting

### Common Issues

1. **Pod Security Policy Violations**
   ```bash
   # Check security context
   kubectl describe pod -l app=templ-pipeline -n your-namespace
   ```

2. **Image Pull Errors**
   ```bash
   # Verify image exists in Harbor
   # Check if namespace has access to Harbor
   kubectl describe pod -l app=templ-pipeline -n your-namespace
   ```

3. **Data Not Found**
   ```bash
   # Check PVC mount
   kubectl exec -it deployment/templ-pipeline -n your-namespace -- ls -la /app/data/
   ```

4. **Ingress Not Working**
   ```bash
   # Check ingress status
   kubectl get ingress -n your-namespace
   kubectl describe ingress templ-pipeline-ingress -n your-namespace
   ```

### Debug Commands

```bash
# Get pod logs
kubectl logs -f deployment/templ-pipeline -n your-namespace

# Execute shell in pod
kubectl exec -it deployment/templ-pipeline -n your-namespace -- /bin/bash

# Check all resources
kubectl get all -l app=templ-pipeline -n your-namespace

# Port forward for local testing
kubectl port-forward svc/templ-pipeline-svc 8501:80 -n your-namespace
```

## Updating the Deployment

### Update Application

```bash
# Build new version
./scripts/build.sh v1.1.0 your-username

# Push to Harbor
docker push hub.cerit.io/your-username/templ-pipeline:v1.1.0

# Update deployment
kubectl set image deployment/templ-pipeline templ-pipeline=hub.cerit.io/your-username/templ-pipeline:v1.1.0 -n your-namespace

# Check rollout
kubectl rollout status deployment/templ-pipeline -n your-namespace
```

### Update Data

```bash
# Copy new data files
./scripts/copy-data.sh your-namespace

# Restart deployment to pick up new data
kubectl rollout restart deployment/templ-pipeline -n your-namespace
```

## Monitoring

### Health Checks

The application includes built-in health checks:
- **Liveness Probe**: `/_stcore/health` (checks if app is running)
- **Readiness Probe**: `/_stcore/health` (checks if app is ready to serve traffic)

### Logs

```bash
# View application logs
kubectl logs -f deployment/templ-pipeline -n your-namespace

# View ingress logs (if needed)
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

## Security Notes

1. **Non-root User**: The container runs as user ID 1000 (non-root)
2. **Read-only Data**: Data volume is mounted read-only
3. **Security Context**: Follows CERIT security policies
4. **TLS**: Automatic HTTPS via Let's Encrypt
5. **Resource Limits**: CPU and memory limits prevent resource exhaustion

## Support

For issues specific to:
- **CERIT Platform**: Contact CERIT support
- **TEMPL Pipeline**: Check project documentation or repository issues
- **Kubernetes**: Refer to Kubernetes documentation

## Cleanup

To remove the deployment:

```bash
# Delete all resources
kubectl delete -f k8s/ -n your-namespace

# Optionally delete PVC (this will delete your data!)
kubectl delete pvc templ-data-pvc -n your-namespace
```



Made changes.

Perfect! I have completed the CERIT deployment setup for your TEMPL Pipeline. Here's a summary of what I've created:

## 🚀 Complete CERIT Deployment Package

### Core Files Created:
1. **docker-entrypoint.sh** - Container startup script with environment configuration
2. **.dockerignore** - Optimized to exclude unnecessary files during build
3. **docker-compose.yml** - Local development setup
4. **docker-compose.prod.yml** - Production configuration

### Kubernetes Manifests (k8s):
1. **`pvc.yaml`** - Persistent Volume Claim for data storage (5Gi)
2. **`deployment.yaml`** - Main application deployment with security contexts
3. **`service.yaml`** - Service to expose the application
4. **`ingress.yaml`** - External access with automatic HTTPS

### Deployment Scripts (scripts):
1. **`build.sh`** - Docker image building for Harbor registry
2. **`deploy.sh`** - Kubernetes deployment automation
3. **`copy-data.sh`** - Data transfer to persistent volume

### Documentation:
1. **CERIT_DEPLOYMENT.md** - Comprehensive deployment guide
2. **deploy-complete.sh** - One-click deployment script

## 🔧 Key Features Implemented:

### Security & Compliance:
- ✅ Non-root user (UID 1000)
- ✅ Security contexts per CERIT requirements
- ✅ Read-only data mounts
- ✅ Resource limits and requests
- ✅ Health checks (liveness/readiness probes)

### CERIT Platform Integration:
- ✅ Harbor registry configuration (`hub.cerit.io`)
- ✅ NFS storage with PVC
- ✅ Ingress with automatic SSL certificates
- ✅ CERIT-compatible domain naming (`*.dyn.cloud.e-infra.cz`)

### Data Management:
- ✅ Persistent storage for ~3GB data files
- ✅ Automated data copying to PVC
- ✅ Read-only data mounts for security

## 🚀 Quick Deployment Steps:

1. **Get Harbor access** at `hub.cerit.io` and note your username
2. **Find your namespace** in the CERIT Rancher dashboard
3. **Choose a unique domain** name (e.g., `templ-yourname`)
4. **Run the complete deployment**:

```bash
# One-command deployment
./deploy-complete.sh -u your-harbor-username -n your-namespace -d templ-yourname
```

This will:
- Build and push your Docker image
- Create all Kubernetes resources
- Copy your data to persistent storage
- Deploy the application with automatic HTTPS

## 🌐 Result:
Your TEMPL Pipeline will be accessible at:
`https://templ-yourname.dyn.cloud.e-infra.cz`

## 📋 Manual Steps (if needed):

If you prefer step-by-step deployment:

```bash
# 1. Login to Harbor
docker login hub.cerit.io

# 2. Build and push
./scripts/build.sh latest your-username
docker push hub.cerit.io/your-username/templ-pipeline:latest

# 3. Update configuration
# Edit k8s/ingress.yaml to change domain name
# Edit k8s/deployment.yaml to update image name

# 4. Deploy
./scripts/copy-data.sh your-namespace
./scripts/deploy.sh your-namespace your-username latest
```

## 🔍 Monitoring & Troubleshooting:

```bash
# Check status
kubectl get pods,svc,ingress -l app=templ-pipeline -n your-namespace

# View logs
kubectl logs -f deployment/templ-pipeline -n your-namespace

# Debug
kubectl describe pod -l app=templ-pipeline -n your-namespace
```

The deployment follows CERIT best practices and your Docker deployment plan. Your Streamlit app will have:
- Automatic HTTPS with Let's Encrypt
- Persistent data storage
- Health monitoring
- Resource limits
- Security compliance
- Easy scaling and updates

Ready to deploy! 🎉