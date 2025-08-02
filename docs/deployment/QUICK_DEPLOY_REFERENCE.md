# TEMPL Pipeline CERIT Deployment - Quick Reference

Quick commands and reference for deploying TEMPL Pipeline to CERIT infrastructure.

## 🚀 One-Command Deployment

```bash
# Complete deployment with all steps
./deploy/scripts/deploy-complete.sh -u your-harbor-username -n your-namespace -d your-domain
```

**Example:**
```bash
./deploy/scripts/deploy-complete.sh -u johndoe -n johndoe-ns -d templ-johndoe
```

## 📋 Required Information

Before deploying, gather these details:

| Item | How to Get | Example |
|------|------------|---------|
| **Harbor Username** | [hub.cerit.io](https://hub.cerit.io/) → User Profile | `johndoe@university.cz` |
| **CLI Secret** | [hub.cerit.io](https://hub.cerit.io/) → User Profile → CLI Secret | `abc123def456` |
| **Namespace** | [rancher.cloud.e-infra.cz](https://rancher.cloud.e-infra.cz/) → Your Project | `johndoe-ns` |
| **Domain Name** | Choose unique name | `templ-johndoe` |

## 🔧 Step-by-Step Commands

### 1. Login to Harbor
```bash
docker login hub.cerit.io
# Enter Harbor username and CLI secret
```

### 2. Build and Push Image
```bash
# Build image
./deploy/scripts/build.sh latest your-username

# Push to Harbor
docker push hub.cerit.io/your-username/templ-pipeline:latest
```

### 3. Update Configuration
```bash
# Update ingress domain
sed -i "s/templ-pipeline\.dyn\.cloud\.e-infra\.cz/your-domain.dyn.cloud.e-infra.cz/g" deploy/kubernetes/ingress.yaml

# Update image reference
sed -i "s|hub.cerit.io/\[your-username\]/templ-pipeline:latest|hub.cerit.io/your-username/templ-pipeline:latest|g" deploy/kubernetes/deployment.yaml
```

### 4. Deploy to Kubernetes
```bash
# Create PVC
kubectl apply -f deploy/kubernetes/pvc.yaml -n your-namespace

# Copy data
./deploy/scripts/copy-data.sh your-namespace

# Deploy application
kubectl apply -f deploy/kubernetes/deployment.yaml -n your-namespace
kubectl apply -f deploy/kubernetes/service.yaml -n your-namespace
kubectl apply -f deploy/kubernetes/ingress.yaml -n your-namespace
```

## 📊 Monitoring Commands

### Check Status
```bash
# View all resources
kubectl get all -l app=templ-pipeline -n your-namespace

# View pod logs
kubectl logs -f deployment/templ-pipeline -n your-namespace

# Check ingress
kubectl get ingress -n your-namespace
```

### Debug Issues
```bash
# Describe pod for errors
kubectl describe pod -l app=templ-pipeline -n your-namespace

# Execute shell in pod
kubectl exec -it deployment/templ-pipeline -n your-namespace -- /bin/bash

# Port forward for local testing
kubectl port-forward svc/templ-pipeline-svc 8501:80 -n your-namespace
```

## 🔄 Update Commands

### Update Application
```bash
# Build new version
./deploy/scripts/build.sh v1.1.0 your-username

# Push to Harbor
docker push hub.cerit.io/your-username/templ-pipeline:v1.1.0

# Update deployment
kubectl set image deployment/templ-pipeline templ-pipeline=hub.cerit.io/your-username/templ-pipeline:v1.1.0 -n your-namespace
```

### Update Data
```bash
# Copy new data
./deploy/scripts/copy-data.sh your-namespace

# Restart to pick up new data
kubectl rollout restart deployment/templ-pipeline -n your-namespace
```

## 🧹 Cleanup Commands

```bash
# Delete all resources
kubectl delete -f deploy/kubernetes/ -n your-namespace

# Delete PVC (removes data!)
kubectl delete pvc templ-data-pvc -n your-namespace
```

## 🌐 Access URLs

Your application will be available at:
```
https://your-domain.dyn.cloud.e-infra.cz
```

**Health Check:**
```
https://your-domain.dyn.cloud.e-infra.cz/_stcore/health
```

## ⚠️ Common Issues

### 1. Image Pull Errors
```bash
# Check if logged in to Harbor
docker login hub.cerit.io

# Verify image exists
docker pull hub.cerit.io/your-username/templ-pipeline:latest
```

### 2. PVC Not Bound
```bash
# Check PVC status
kubectl get pvc -n your-namespace

# Wait for binding
kubectl wait --for=condition=Bound pvc/templ-data-pvc -n your-namespace --timeout=120s
```

### 3. Pod Not Starting
```bash
# Check pod events
kubectl describe pod -l app=templ-pipeline -n your-namespace

# Check logs
kubectl logs deployment/templ-pipeline -n your-namespace
```

### 4. Ingress Not Working
```bash
# Check ingress status
kubectl get ingress -n your-namespace

# Check certificate
kubectl describe ingress templ-pipeline-ingress -n your-namespace
```

## 📞 Support Resources

- **CERIT Documentation**: https://docs.cerit.io/
- **Harbor Registry**: https://hub.cerit.io/
- **Rancher Dashboard**: https://rancher.cloud.e-infra.cz/
- **e-INFRA Cloud**: https://docs.e-infra.cz/cloud/einfracz-cloud/

## ✅ Quick Checklist

- [ ] Harbor login successful
- [ ] Image built and pushed
- [ ] Configuration files updated
- [ ] PVC created and bound
- [ ] Data copied to PVC
- [ ] Application deployed
- [ ] Ingress configured
- [ ] SSL certificate issued
- [ ] Application accessible via HTTPS

## 🎯 Result

After successful deployment:
- **Web Interface**: Available at `https://your-domain.dyn.cloud.e-infra.cz`
- **CLI Tools**: Available via kubectl exec
- **Data Storage**: Persistent volume with molecular data
- **Security**: HTTPS with automatic certificate renewal
- **Monitoring**: Health checks and logging enabled 