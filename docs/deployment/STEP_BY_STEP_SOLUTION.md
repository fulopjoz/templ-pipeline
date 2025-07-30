# TEMPL Pipeline CERIT Deployment - Complete Step-by-Step Solution

This guide provides a complete solution for deploying your TEMPL Pipeline web application to the CERIT infrastructure, updated for the latest repository changes.

## 🎯 What You'll Deploy

Your TEMPL Pipeline includes:
- **Streamlit Web Interface**: Interactive UI for protein-ligand pose prediction
- **CLI Tools**: Command-line interface for batch processing
- **Scientific Computing**: RDKit, Biopython, and other scientific libraries
- **Data Management**: Persistent storage for molecular data (~3GB)
- **Security**: HTTPS with automatic certificate renewal

## 📋 Prerequisites Checklist

Before starting, ensure you have:

### 1. CERIT Access
- [ ] Academic affiliation with access to CERIT-SC platform
- [ ] Access to Harbor registry at `hub.cerit.io`
- [ ] Kubernetes namespace assigned in CERIT
- [ ] Domain name chosen (unique name ending with `.dyn.cloud.e-infra.cz`)

### 2. Local Environment
- [ ] Docker installed and running
- [ ] kubectl configured for CERIT cluster
- [ ] Git repository cloned locally

### 3. Required Information
- [ ] Harbor username (from `hub.cerit.io`)
- [ ] Harbor CLI secret (from Harbor profile)
- [ ] Kubernetes namespace (from Rancher dashboard)
- [ ] Unique domain name (e.g., `templ-yourname`)

## 🚀 Solution: Complete Deployment Process

### Phase 1: Environment Setup

#### Step 1.1: Get Your CERIT Information

**A. Harbor Registry Access**
1. Go to [https://hub.cerit.io/](https://hub.cerit.io/)
2. Login with your academic credentials
3. Note your **Harbor Username** (usually your email)
4. Go to User Profile → CLI Secret and copy your **CLI Secret**

**B. Kubernetes Namespace**
1. Go to [https://rancher.cloud.e-infra.cz/](https://rancher.cloud.e-infra.cz/)
2. Login with your academic credentials
3. Note your **Namespace** (usually `lastname-ns` or similar)

**C. Choose Domain Name**
- Pick a unique name like `templ-yourname`
- Full domain will be: `templ-yourname.dyn.cloud.e-infra.cz`

#### Step 1.2: Prepare Local Environment

```bash
# Ensure you're in the project directory
cd /path/to/templ_pipeline

# Verify you have the latest version
git pull origin main

# Check that deployment files exist
ls -la deploy/scripts/
ls -la deploy/kubernetes/
ls -la deploy/docker/
```

#### Step 1.3: Login to Harbor Registry

```bash
# Login to Harbor
docker login hub.cerit.io
# Enter your Harbor username and CLI secret when prompted

# Verify login
docker info | grep "Registry"
```

### Phase 2: Build and Push Docker Image

#### Step 2.1: Build the Docker Image

```bash
# Build the image with your Harbor username
./deploy/scripts/build.sh latest your-harbor-username

# Verify the image was built
docker images | grep templ-pipeline
```

**Expected Output:**
```
REPOSITORY                                    TAG       IMAGE ID       CREATED         SIZE
hub.cerit.io/your-username/templ-pipeline   latest    abc123def456   2 minutes ago   2.1GB
```

#### Step 2.2: Push to Harbor Registry

```bash
# Push the image to Harbor
docker push hub.cerit.io/your-harbor-username/templ-pipeline:latest

# Verify the image is in Harbor
# Go to https://hub.cerit.io/ and check your repository
```

### Phase 3: Update Configuration Files

#### Step 3.1: Update Ingress Domain

Edit `deploy/kubernetes/ingress.yaml`:

```yaml
# Find this section:
spec:
  tls:
    - hosts:
        - "templ-pipeline.dyn.cloud.e-infra.cz"  # CHANGE THIS
      secretName: templ-pipeline-dyn-cloud-e-infra-cz-tls
  rules:
  - host: "templ-pipeline.dyn.cloud.e-infra.cz"  # CHANGE THIS

# Change to your domain:
spec:
  tls:
    - hosts:
        - "your-unique-name.dyn.cloud.e-infra.cz"  # YOUR DOMAIN
      secretName: your-unique-name-dyn-cloud-e-infra-cz-tls
  rules:
  - host: "your-unique-name.dyn.cloud.e-infra.cz"  # YOUR DOMAIN
```

#### Step 3.2: Update Image Reference

Edit `deploy/kubernetes/deployment.yaml`:

```yaml
# Find this line:
image: hub.cerit.io/[your-username]/templ-pipeline:latest

# Change to your Harbor username:
image: hub.cerit.io/your-actual-harbor-username/templ-pipeline:latest
```

### Phase 4: Deploy to Kubernetes

#### Step 4.1: Create Persistent Volume Claim

```bash
# Create PVC for data storage
kubectl apply -f deploy/kubernetes/pvc.yaml -n your-namespace

# Wait for PVC to be bound
kubectl wait --for=condition=Bound pvc/templ-data-pvc -n your-namespace --timeout=120s

# Verify PVC status
kubectl get pvc -n your-namespace
```

**Expected Output:**
```
NAME             STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
templ-data-pvc   Bound    pvc-abc123-def456-ghi789-jkl012-mno345   5Gi        RWO            nfs-csi        30s
```

#### Step 4.2: Copy Data to Persistent Volume

```bash
# Copy data files to PVC
./deploy/scripts/copy-data.sh your-namespace

# Verify data was copied (optional)
kubectl run --rm -it --restart=Never data-check --image=alpine:latest \
  --overrides='{"spec":{"securityContext":{"runAsUser":1000,"runAsNonRoot":true},"containers":[{"name":"data-check","image":"alpine:latest","command":["sh"],"securityContext":{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]}},"volumeMounts":[{"mountPath":"/data","name":"data-volume"}]}],"volumes":[{"name":"data-volume","persistentVolumeClaim":{"claimName":"templ-data-pvc"}}]}}' \
  -n your-namespace -- ls -la /data/
```

#### Step 4.3: Deploy Application

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

**Expected Output:**
```
NAME                                READY   STATUS    RESTARTS   AGE
pod/templ-pipeline-abc123-def456   1/1     Running   0          2m

NAME                        TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
service/templ-pipeline-svc  ClusterIP   10.96.123.45   <none>        80/TCP    2m

NAME                                    CLASS   HOSTS                                    ADDRESS         PORTS     AGE
ingress.networking.k8s.io/templ-pipeline-ingress   nginx   your-unique-name.dyn.cloud.e-infra.cz   123.45.67.89   80, 443   2m
```

### Phase 5: Verify Deployment

#### Step 5.1: Check Application Status

```bash
# View all resources
kubectl get all -l app=templ-pipeline -n your-namespace

# View pod logs
kubectl logs -f deployment/templ-pipeline -n your-namespace

# Check ingress status
kubectl get ingress -n your-namespace
kubectl describe ingress templ-pipeline-ingress -n your-namespace
```

#### Step 5.2: Test Application Access

```bash
# Test health endpoint
kubectl port-forward svc/templ-pipeline-svc 8080:80 -n your-namespace
# Then visit http://localhost:8080/_stcore/health

# Or test directly via ingress (after DNS propagation)
curl -I https://your-unique-name.dyn.cloud.e-infra.cz/_stcore/health
```

## 🌐 Access Your Application

Your TEMPL Pipeline will be available at:
```
https://your-unique-name.dyn.cloud.e-infra.cz
```

**Important Notes:**
- SSL certificate issuance may take 5-10 minutes
- DNS propagation may take a few minutes
- Ingress controller routing may take 1-2 minutes

## 📊 Monitoring and Troubleshooting

### Check Application Health

```bash
# View real-time logs
kubectl logs -f deployment/templ-pipeline -n your-namespace

# Check pod status
kubectl describe pod -l app=templ-pipeline -n your-namespace

# Test health endpoint
curl https://your-unique-name.dyn.cloud.e-infra.cz/_stcore/health
```

### Common Issues and Solutions

#### Issue 1: Pod Not Starting
```bash
# Check pod events
kubectl describe pod -l app=templ-pipeline -n your-namespace

# Check if image exists
docker pull hub.cerit.io/your-username/templ-pipeline:latest
```

#### Issue 2: PVC Not Bound
```bash
# Check PVC status
kubectl get pvc -n your-namespace

# Wait for binding
kubectl wait --for=condition=Bound pvc/templ-data-pvc -n your-namespace --timeout=120s
```

#### Issue 3: Ingress Not Working
```bash
# Check ingress status
kubectl get ingress -n your-namespace

# Check certificate
kubectl describe ingress templ-pipeline-ingress -n your-namespace
```

#### Issue 4: Data Not Found
```bash
# Check data mount
kubectl exec -it deployment/templ-pipeline -n your-namespace -- ls -la /app/data/

# Re-copy data if needed
./deploy/scripts/copy-data.sh your-namespace
```

## 🔄 Updating Your Deployment

### Update Application Code

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

### Update Data Files

```bash
# Copy new data files
./deploy/scripts/copy-data.sh your-namespace

# Restart deployment to pick up new data
kubectl rollout restart deployment/templ-pipeline -n your-namespace
```

## 🧹 Cleanup

To remove your deployment:

```bash
# Delete all resources
kubectl delete -f deploy/kubernetes/ -n your-namespace

# Optionally delete PVC (this will delete your data!)
kubectl delete pvc templ-data-pvc -n your-namespace
```

## ✅ Success Verification

After deployment, verify:

- [ ] Application accessible at `https://your-unique-name.dyn.cloud.e-infra.cz`
- [ ] SSL certificate issued (green lock in browser)
- [ ] Streamlit interface loads correctly
- [ ] Health endpoint responds: `https://your-unique-name.dyn.cloud.e-infra.cz/_stcore/health`
- [ ] Pod logs show no errors
- [ ] Data files accessible in `/app/data/`

## 🎉 What You've Deployed

Your TEMPL Pipeline now provides:

1. **Web Interface**: Streamlit-based UI for protein-ligand pose prediction
2. **CLI Tools**: Command-line interface for batch processing
3. **Data Management**: Persistent storage for molecular data
4. **Security**: HTTPS with automatic certificate renewal
5. **Monitoring**: Health checks and comprehensive logging
6. **Scalability**: Kubernetes-based deployment with resource limits
7. **Scientific Computing**: RDKit, Biopython, and other scientific libraries

## 📞 Support Resources

- **CERIT Documentation**: https://docs.cerit.io/
- **Harbor Registry**: https://hub.cerit.io/
- **Rancher Dashboard**: https://rancher.cloud.e-infra.cz/
- **e-INFRA Cloud**: https://docs.e-infra.cz/cloud/einfracz-cloud/
- **TEMPL Pipeline Issues**: https://github.com/fulopjoz/templ-pipeline/issues

Your TEMPL Pipeline is now ready for scientific research and collaboration! 🚀 