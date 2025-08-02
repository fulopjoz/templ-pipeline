# TEMPL Pipeline CERIT Deployment Guide - Updated

This guide provides step-by-step instructions for deploying the TEMPL Pipeline on the CERIT-SC Kubernetes platform, updated for the latest repository structure.

## 🚀 Quick Start (One-Command Deployment)

If you have all your CERIT information ready:

```bash
# One-command deployment
./deploy/scripts/deploy-complete.sh -u your-harbor-username -n your-namespace -d your-unique-domain
```

**Example:**
```bash
./deploy/scripts/deploy-complete.sh -u johndoe -n johndoe-ns -d templ-johndoe
```

This will deploy your app to: `https://templ-johndoe.dyn.cloud.e-infra.cz`

## 📋 Prerequisites

### 1. CERIT Access Requirements
- **Academic Affiliation**: You need access to the CERIT-SC platform
- **Harbor Account**: Access to Harbor registry at `hub.cerit.io`
- **Kubernetes Namespace**: Your assigned namespace in CERIT
- **Domain Name**: Choose a unique name ending with `.dyn.cloud.e-infra.cz`

### 2. Local Environment Setup
```bash
# Install required tools
sudo apt-get update
sudo apt-get install -y docker.io kubectl curl

# Verify installations
docker --version
kubectl version --client
```

### 3. Get Your CERIT Information

#### A. Harbor Registry Access
1. Go to [https://hub.cerit.io/](https://hub.cerit.io/)
2. Login with your academic credentials
3. Note your **Harbor Username** (usually your email or username)
4. Get your **CLI Secret** from User Profile → CLI Secret

#### B. Kubernetes Namespace
1. Go to [https://rancher.cloud.e-infra.cz/](https://rancher.cloud.e-infra.cz/)
2. Login with your academic credentials
3. Note your **Namespace** (usually `lastname-ns` or similar)

#### C. Choose Domain Name
- Pick a unique name like `templ-yourname`
- Full domain will be: `templ-yourname.dyn.cloud.e-infra.cz`

## 🔧 Step-by-Step Deployment

### Step 1: Prepare Your Environment

```bash
# Clone the repository (if not already done)
git clone https://github.com/fulopjoz/templ-pipeline.git
cd templ-pipeline

# Ensure you're on the latest version
git pull origin main
```

### Step 2: Login to Harbor Registry

```bash
# Login to Harbor
docker login hub.cerit.io
# Enter your Harbor username and CLI secret when prompted
```

### Step 3: Update Configuration Files

#### A. Update Ingress Domain
Edit `deploy/kubernetes/ingress.yaml`:
```yaml
# Change this line:
- "templ-pipeline.dyn.cloud.e-infra.cz"
# To your domain:
- "your-unique-name.dyn.cloud.e-infra.cz"
```

#### B. Update Image Reference
Edit `deploy/kubernetes/deployment.yaml`:
```yaml
# Change this line:
image: hub.cerit.io/[your-username]/templ-pipeline:latest
# To your Harbor username:
image: hub.cerit.io/your-actual-username/templ-pipeline:latest
```

### Step 4: Build Docker Image

```bash
# Build the image
./deploy/scripts/build.sh latest your-harbor-username

# Verify the image was built
docker images | grep templ-pipeline
```

### Step 5: Push to Harbor Registry

```bash
# Push the image to Harbor
docker push hub.cerit.io/your-harbor-username/templ-pipeline:latest

# Verify the image is in Harbor
# Go to https://hub.cerit.io/ and check your repository
```

### Step 6: Create Kubernetes Resources

```bash
# Create Persistent Volume Claim
kubectl apply -f deploy/kubernetes/pvc.yaml -n your-namespace

# Wait for PVC to be bound
kubectl wait --for=condition=Bound pvc/templ-data-pvc -n your-namespace --timeout=120s

# Check PVC status
kubectl get pvc -n your-namespace
```

### Step 7: Copy Data to Persistent Volume

```bash
# Copy data files to PVC
./deploy/scripts/copy-data.sh your-namespace

# Verify data was copied (optional)
kubectl run --rm -it --restart=Never data-check --image=alpine:latest \
  --overrides='{"spec":{"securityContext":{"runAsUser":1000,"runAsNonRoot":true},"containers":[{"name":"data-check","image":"alpine:latest","command":["sh"],"securityContext":{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]}},"volumeMounts":[{"mountPath":"/data","name":"data-volume"}]}],"volumes":[{"name":"data-volume","persistentVolumeClaim":{"claimName":"templ-data-pvc"}}]}}' \
  -n your-namespace -- ls -la /data/
```

### Step 8: Deploy Application

```bash
# Deploy all Kubernetes resources
kubectl apply -f deploy/kubernetes/deployment.yaml -n your-namespace
kubectl apply -f deploy/kubernetes/service.yaml -n your-namespace
kubectl apply -f deploy/kubernetes/ingress.yaml -n your-namespace

# Wait for deployment to be ready
kubectl wait --for=condition=available --timeout=300s deployment/templ-pipeline -n your-namespace

# Check deployment status
kubectl get pods,svc,ingress -l app=templ-pipeline -n your-namespace
```

### Step 9: Verify Deployment

```bash
# Check pod logs
kubectl logs -f deployment/templ-pipeline -n your-namespace

# Check pod details
kubectl describe pod -l app=templ-pipeline -n your-namespace

# Test health endpoint
kubectl port-forward svc/templ-pipeline-svc 8080:80 -n your-namespace
# Then visit http://localhost:8080/_stcore/health
```

## 🌐 Access Your Application

Your TEMPL Pipeline will be available at:
```
https://your-unique-name.dyn.cloud.e-infra.cz
```

**Note**: It may take 5-10 minutes for:
- SSL certificate to be issued by Let's Encrypt
- DNS propagation
- Ingress controller to route traffic

## 📊 Monitoring & Troubleshooting

### Check Application Status

```bash
# View all resources
kubectl get all -l app=templ-pipeline -n your-namespace

# View pod logs
kubectl logs -f deployment/templ-pipeline -n your-namespace

# Check ingress status
kubectl get ingress -n your-namespace
kubectl describe ingress templ-pipeline-ingress -n your-namespace
```

### Common Issues & Solutions

#### 1. Pod Security Policy Violations
```bash
# Check security context
kubectl describe pod -l app=templ-pipeline -n your-namespace
```

#### 2. Image Pull Errors
```bash
# Verify image exists in Harbor
# Check if namespace has access to Harbor
kubectl describe pod -l app=templ-pipeline -n your-namespace
```

#### 3. Data Not Found
```bash
# Check PVC mount
kubectl exec -it deployment/templ-pipeline -n your-namespace -- ls -la /app/data/
```

#### 4. Ingress Not Working
```bash
# Check ingress status
kubectl get ingress -n your-namespace
kubectl describe ingress templ-pipeline-ingress -n your-namespace
```

### Debug Commands

```bash
# Execute shell in pod
kubectl exec -it deployment/templ-pipeline -n your-namespace -- /bin/bash

# Port forward for local testing
kubectl port-forward svc/templ-pipeline-svc 8501:80 -n your-namespace

# Check all resources
kubectl get all -l app=templ-pipeline -n your-namespace
```

## 🔄 Updating the Deployment

### Update Application

```bash
# Build new version
./deploy/scripts/build.sh v1.1.0 your-username

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
./deploy/scripts/copy-data.sh your-namespace

# Restart deployment to pick up new data
kubectl rollout restart deployment/templ-pipeline -n your-namespace
```

## 🗂️ File Structure

```
templ_pipeline/
├── deploy/                          # Deployment files
│   ├── docker/                      # Docker configuration
│   │   ├── Dockerfile               # Multi-stage Docker build
│   │   ├── docker-entrypoint.sh    # Container entry point
│   │   ├── docker-compose.yml      # Local development
│   │   └── docker-compose.prod.yml # Production compose
│   ├── kubernetes/                  # Kubernetes manifests
│   │   ├── deployment.yaml         # Main application deployment
│   │   ├── service.yaml            # Service definition
│   │   ├── ingress.yaml            # Ingress for external access
│   │   └── pvc.yaml                # Persistent volume claim
│   └── scripts/                     # Deployment scripts
│       ├── build.sh                # Docker build script
│       ├── deploy.sh               # Kubernetes deployment
│       ├── copy-data.sh            # Data copy utility
│       └── deploy-complete.sh      # One-command deployment
├── templ_pipeline/                  # Application source code
│   ├── ui/                         # Streamlit web interface
│   │   └── app.py                  # Main Streamlit application
│   └── cli/                        # Command-line interface
├── scripts/                         # Utility scripts
│   └── run_streamlit_app.py        # Streamlit launcher
├── requirements.txt                 # Python dependencies
└── pyproject.toml                  # Project configuration
```

## 📈 Resource Requirements

### Minimum Requirements
- **CPU**: 500m (0.5 cores)
- **Memory**: 2Gi
- **Storage**: 5Gi for data + 2Gi for temp files

### Recommended for Production
- **CPU**: 2 cores
- **Memory**: 4Gi
- **Storage**: 10Gi for data + 5Gi for temp files

## 🔒 Security Features

1. **Non-root User**: Container runs as user ID 1000
2. **Read-only Data**: Data volume mounted read-only
3. **Security Context**: Follows CERIT security policies
4. **TLS**: Automatic HTTPS via Let's Encrypt
5. **Resource Limits**: CPU and memory limits prevent resource exhaustion

## 🧹 Cleanup

To remove the deployment:

```bash
# Delete all resources
kubectl delete -f deploy/kubernetes/ -n your-namespace

# Optionally delete PVC (this will delete your data!)
kubectl delete pvc templ-data-pvc -n your-namespace
```

## 📞 Support

For issues specific to:
- **CERIT Platform**: Contact CERIT support
- **TEMPL Pipeline**: Check project documentation or repository issues
- **Kubernetes**: Refer to Kubernetes documentation

## 🔗 Useful Links

- **CERIT Documentation**: https://docs.cerit.io/
- **Harbor Registry**: https://hub.cerit.io/
- **Rancher Dashboard**: https://rancher.cloud.e-infra.cz/
- **e-INFRA Cloud**: https://docs.e-infra.cz/cloud/einfracz-cloud/

## ✅ Deployment Checklist

- [ ] CERIT access configured
- [ ] Harbor account created and CLI secret obtained
- [ ] Kubernetes namespace identified
- [ ] Unique domain name chosen
- [ ] Docker image built and pushed to Harbor
- [ ] Kubernetes resources created
- [ ] Data copied to persistent volume
- [ ] Application deployed and running
- [ ] Ingress configured and accessible
- [ ] SSL certificate issued
- [ ] Application accessible via HTTPS

## 🎉 Success!

Once deployed, your TEMPL Pipeline will provide:
- **Web Interface**: Streamlit-based UI for protein-ligand pose prediction
- **CLI Tools**: Command-line interface for batch processing
- **Data Management**: Persistent storage for molecular data
- **Security**: HTTPS with automatic certificate renewal
- **Monitoring**: Health checks and logging
- **Scalability**: Kubernetes-based deployment

Your application is now ready for scientific research and collaboration! 🚀 