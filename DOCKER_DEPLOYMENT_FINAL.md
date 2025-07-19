# TEMPL Pipeline Docker Deployment to CERIT - Complete Guide

This is the complete step-by-step guide for deploying the TEMPL Pipeline Streamlit web application to the CERIT-SC Kubernetes platform using Docker containers.

## 📋 Complete List of Files Created

### Core Docker Files
```
templ_pipeline/
├── Dockerfile                      # ✅ Multi-stage Docker build (already exists)
├── docker-entrypoint.sh           # ✅ Container startup script
├── .dockerignore                  # ✅ Docker build exclusions
├── docker-compose.yml             # ✅ Local development setup
└── docker-compose.prod.yml        # ✅ Production configuration
```

### Kubernetes Deployment Files
```
k8s/
├── deployment.yaml                 # ✅ Main application deployment
├── service.yaml                   # ✅ Service definition
├── ingress.yaml                   # ✅ External access with HTTPS
└── pvc.yaml                       # ✅ Persistent volume claim for data
```

### Deployment Scripts
```
scripts/
├── build.sh                       # ✅ Docker build automation
├── deploy.sh                      # ✅ Kubernetes deployment automation
└── copy-data.sh                   # ✅ Data transfer utility
```

### Documentation
```
├── CERIT_DEPLOYMENT.md             # ✅ Detailed deployment guide
└── DOCKER_DEPLOYMENT_FINAL.md     # ✅ This file - step-by-step guide
```

## 🔧 Required Changes Before Deployment

### 1. Get Your CERIT Information

You need to collect this information before starting:

- **Harbor Username**: Your username at `hub.cerit.io`
- **Harbor CLI Secret**: From your profile at `hub.cerit.io` → User Profile → CLI Secret
- **Kubernetes Namespace**: From CERIT Rancher dashboard (usually `lastname-ns`)
- **Desired Domain Name**: Choose a unique name ending with `.dyn.cloud.e-infra.cz`

### 2. Mandatory File Changes

#### A. Update Kubernetes Deployment Image Reference

**File**: `k8s/deployment.yaml`
**Line**: ~35
**Change**:
```yaml
# FROM:
image: hub.cerit.io/[your-username]/templ-pipeline:latest

# TO: (replace 'your-actual-username' with your Harbor username)
image: hub.cerit.io/your-actual-username/templ-pipeline:latest
```

#### B. Update Ingress Domain Name

**File**: `k8s/ingress.yaml`
**Lines**: ~14 and ~18
**Change**:
```yaml
# FROM:
- "templ-pipeline.dyn.cloud.e-infra.cz"

# TO: (choose a unique name)
- "templ-yourname.dyn.cloud.e-infra.cz"
```

**Also update the secretName**:
```yaml
# FROM:
secretName: templ-pipeline-dyn-cloud-e-infra-cz-tls

# TO: (match your domain with dots replaced by dashes)
secretName: templ-yourname-dyn-cloud-e-infra-cz-tls
```

#### C. Update Build Script (Optional)

**File**: `scripts/build.sh`
**Line**: ~5
**Change the default username**:
```bash
# FROM:
HARBOR_USERNAME=${2:-$USER}

# TO: (replace with your Harbor username)
HARBOR_USERNAME=${2:-your-actual-username}
```

## 🚀 Step-by-Step Deployment Process

### Step 1: Access CERIT and Harbor

1. **Login to CERIT Rancher Dashboard**:
   - Go to: https://rancher.cloud.e-infra.cz/
   - Login with your academic credentials
   - Note your namespace name (usually `lastname-ns`)

2. **Login to Harbor Registry**:
   - Go to: https://hub.cerit.io/
   - Login with your academic credentials
   - Go to User Profile → copy your CLI Secret

3. **Configure kubectl** (if not already done):
   ```bash
   # Download kubeconfig from Rancher dashboard
   # Move it to ~/.kube/config or set KUBECONFIG environment variable
   kubectl cluster-info  # Test connection
   ```

### Step 2: Prepare Local Environment

1. **Navigate to project directory**:
   ```bash
   cd /home/ubuntu/mcs/templ_pipeline
   ```

2. **Verify all files exist**:
   ```bash
   ls -la docker-entrypoint.sh k8s/ scripts/
   ls -la Dockerfile docker-compose.yml .dockerignore
   ```

3. **Make scripts executable** (if not already):
   ```bash
   chmod +x docker-entrypoint.sh scripts/*.sh
   ```

### Step 3: Customize Configuration Files

1. **Edit Kubernetes deployment**:
   ```bash
   nano k8s/deployment.yaml
   # Replace [your-username] with your Harbor username
   ```

2. **Edit Ingress configuration**:
   ```bash
   nano k8s/ingress.yaml
   # Replace templ-pipeline with your unique name
   # Update both the host and secretName fields
   ```

3. **Test domain availability**:
   ```bash
   # Check if your chosen domain is available
   curl https://your-chosen-name.dyn.cloud.e-infra.cz
   # Should return "connection refused" or similar (not an existing page)
   ```

### Step 4: Build and Push Docker Image

1. **Login to Harbor**:
   ```bash
   docker login hub.cerit.io
   # Username: your-harbor-username
   # Password: your-CLI-secret-from-harbor-profile
   ```

2. **Build the Docker image**:
   ```bash
   ./scripts/build.sh latest your-harbor-username
   ```
   
   Or manually:
   ```bash
   docker build -t hub.cerit.io/your-harbor-username/templ-pipeline:latest .
   ```

3. **Push to Harbor**:
   ```bash
   docker push hub.cerit.io/your-harbor-username/templ-pipeline:latest
   ```

4. **Verify image in Harbor**:
   - Go to https://hub.cerit.io/
   - Check your project → repositories
   - Confirm `templ-pipeline` image exists

### Step 5: Create Persistent Storage and Copy Data

1. **Create the PVC**:
   ```bash
   kubectl apply -f k8s/pvc.yaml -n your-namespace
   ```

2. **Wait for PVC to be bound**:
   ```bash
   kubectl wait --for=condition=Bound pvc/templ-data-pvc -n your-namespace --timeout=60s
   kubectl get pvc -n your-namespace
   ```

3. **Copy data to persistent volume**:
   ```bash
   ./scripts/copy-data.sh your-namespace
   ```

   Or manually:
   ```bash
   # Create temporary pod for data copy
   kubectl run data-copy --image=alpine:latest --rm -it --restart=Never \
     --overrides='{"spec":{"securityContext":{"runAsUser":1000,"runAsNonRoot":true},"containers":[{"name":"data-copy","image":"alpine:latest","command":["sleep","3600"],"securityContext":{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]}},"volumeMounts":[{"mountPath":"/data","name":"data-volume"}]}],"volumes":[{"name":"data-volume","persistentVolumeClaim":{"claimName":"templ-data-pvc"}}]}}' \
     -n your-namespace

   # In another terminal, copy data
   kubectl cp ./data your-namespace/data-copy:/data/ --no-preserve=true

   # Exit and delete the pod
   kubectl delete pod data-copy -n your-namespace
   ```

4. **Verify data copy**:
   ```bash
   kubectl run data-check --image=alpine:latest --rm -it --restart=Never \
     --overrides='{"spec":{"securityContext":{"runAsUser":1000,"runAsNonRoot":true},"containers":[{"name":"data-check","image":"alpine:latest","command":["sh"],"securityContext":{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]}},"volumeMounts":[{"mountPath":"/data","name":"data-volume"}]}],"volumes":[{"name":"data-volume","persistentVolumeClaim":{"claimName":"templ-data-pvc"}}]}}' \
     -n your-namespace \
     -- ls -la /data/
   ```

### Step 6: Deploy Application to Kubernetes

1. **Deploy all Kubernetes resources**:
   ```bash
   kubectl apply -f k8s/ -n your-namespace
   ```

   Or deploy individually:
   ```bash
   kubectl apply -f k8s/deployment.yaml -n your-namespace
   kubectl apply -f k8s/service.yaml -n your-namespace
   kubectl apply -f k8s/ingress.yaml -n your-namespace
   ```

2. **Monitor deployment**:
   ```bash
   # Watch pod creation
   kubectl get pods -l app=templ-pipeline -n your-namespace -w

   # Check deployment status
   kubectl rollout status deployment/templ-pipeline -n your-namespace

   # Check all resources
   kubectl get pods,svc,ingress -l app=templ-pipeline -n your-namespace
   ```

### Step 7: Verify and Access Application

1. **Check pod logs**:
   ```bash
   kubectl logs -f deployment/templ-pipeline -n your-namespace
   ```

2. **Verify health check**:
   ```bash
   # Port forward to test locally
   kubectl port-forward svc/templ-pipeline-svc 8080:80 -n your-namespace &
   curl http://localhost:8080/_stcore/health
   ```

3. **Wait for DNS propagation** (1-2 minutes):
   ```bash
   # Check if domain is accessible
   curl -I https://your-chosen-name.dyn.cloud.e-infra.cz
   ```

4. **Access your application**:
   - Open browser: `https://your-chosen-name.dyn.cloud.e-infra.cz`
   - You should see the TEMPL Pipeline Streamlit interface

## 🔍 Troubleshooting Common Issues

### Pod Won't Start

**Check security context**:
```bash
kubectl describe pod -l app=templ-pipeline -n your-namespace
```

**Common fixes**:
- Ensure `runAsUser: 1000` and `runAsNonRoot: true`
- Check image pull permissions

### Image Pull Errors

**Check image exists**:
```bash
docker pull hub.cerit.io/your-username/templ-pipeline:latest
```

**Verify Harbor login**:
```bash
docker login hub.cerit.io
```

### Ingress Not Working

**Check ingress status**:
```bash
kubectl get ingress -n your-namespace
kubectl describe ingress templ-pipeline-ingress -n your-namespace
```

**Common issues**:
- Domain name already in use
- Invalid domain format
- DNS propagation delay (wait 2-3 minutes)

### Data Not Found

**Check data mount**:
```bash
kubectl exec -it deployment/templ-pipeline -n your-namespace -- ls -la /app/data/
```

**Re-copy data if needed**:
```bash
./scripts/copy-data.sh your-namespace
```

## 📊 Resource Monitoring

### Check Resource Usage
```bash
# Pod resource usage
kubectl top pod -l app=templ-pipeline -n your-namespace

# Node resource usage
kubectl top nodes

# Describe pod for detailed info
kubectl describe pod -l app=templ-pipeline -n your-namespace
```

### Scale Application (if needed)
```bash
# Scale to 2 replicas
kubectl scale deployment templ-pipeline --replicas=2 -n your-namespace

# Check scaling
kubectl get pods -l app=templ-pipeline -n your-namespace
```

## 🔄 Updating the Application

### Update Application Code
```bash
# Build new version
./scripts/build.sh v1.1.0 your-username

# Push to Harbor
docker push hub.cerit.io/your-username/templ-pipeline:v1.1.0

# Update deployment
kubectl set image deployment/templ-pipeline templ-pipeline=hub.cerit.io/your-username/templ-pipeline:v1.1.0 -n your-namespace

# Monitor rollout
kubectl rollout status deployment/templ-pipeline -n your-namespace
```

### Update Data
```bash
# Copy new data
./scripts/copy-data.sh your-namespace

# Restart pods to pick up new data
kubectl rollout restart deployment/templ-pipeline -n your-namespace
```

## 🧹 Cleanup

### Remove Application
```bash
# Delete all resources except PVC
kubectl delete deployment,service,ingress -l app=templ-pipeline -n your-namespace
```

### Complete Cleanup (including data)
```bash
# Delete everything including data
kubectl delete -f k8s/ -n your-namespace

# This will delete your data permanently!
```

## 📞 Support and Resources

### CERIT Documentation
- Platform: https://docs.cerit.io/
- Harbor: https://docs.cerit.io/en/docs/docker/harbor
- Kubernetes: https://docs.cerit.io/en/docs/examples/helloworld

### Useful Commands Reference
```bash
# Quick status check
kubectl get all -l app=templ-pipeline -n your-namespace

# Debug pod issues
kubectl describe pod -l app=templ-pipeline -n your-namespace
kubectl logs -f deployment/templ-pipeline -n your-namespace

# Port forward for testing
kubectl port-forward svc/templ-pipeline-svc 8501:80 -n your-namespace

# Execute shell in pod
kubectl exec -it deployment/templ-pipeline -n your-namespace -- /bin/bash

# Check resource usage
kubectl top pod -l app=templ-pipeline -n your-namespace
```

## ✅ Deployment Checklist

Before deployment:
- [ ] CERIT access configured
- [ ] Harbor username and CLI secret obtained
- [ ] kubectl configured for CERIT cluster
- [ ] Namespace name identified
- [ ] Unique domain name chosen
- [ ] Files customized with your information

During deployment:
- [ ] Docker image built and pushed to Harbor
- [ ] PVC created and bound
- [ ] Data copied to persistent volume
- [ ] Kubernetes resources deployed
- [ ] Pod started successfully
- [ ] Ingress configured and accessible

After deployment:
- [ ] Application accessible via HTTPS
- [ ] Health checks working
- [ ] Data files loaded correctly
- [ ] Performance acceptable

## 🎉 Success!

If everything worked correctly, your TEMPL Pipeline should now be running on CERIT at:
`https://your-chosen-name.dyn.cloud.e-infra.cz`

The application features:
- ✅ Automatic HTTPS with Let's Encrypt
- ✅ Persistent data storage (~3GB)
- ✅ Health monitoring and auto-restart
- ✅ Resource limits and security compliance
- ✅ Scalable architecture
- ✅ Professional deployment following CERIT best practices
