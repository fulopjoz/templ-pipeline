# TEMPL Pipeline CERIT Deployment - Quick Commands

## 🚀 One-Line Deployment Commands

### Prerequisites Setup
```bash
# 1. Get your info from CERIT
HARBOR_USERNAME="your-harbor-username"    # From hub.cerit.io
NAMESPACE="your-namespace"                # From Rancher (usually lastname-ns)
DOMAIN="your-unique-name"                 # Choose unique name for .dyn.cloud.e-infra.cz
```

### Quick Deployment
```bash
# 1. Login to Harbor
docker login hub.cerit.io

# 2. Update configuration files
sed -i "s/\[your-username\]/$HARBOR_USERNAME/g" k8s/deployment.yaml
sed -i "s/templ-pipeline/$DOMAIN/g" k8s/ingress.yaml

# 3. Build and push
docker build -t hub.cerit.io/$HARBOR_USERNAME/templ-pipeline:latest .
docker push hub.cerit.io/$HARBOR_USERNAME/templ-pipeline:latest

# 4. Deploy to Kubernetes
kubectl apply -f k8s/pvc.yaml -n $NAMESPACE
kubectl wait --for=condition=Bound pvc/templ-data-pvc -n $NAMESPACE --timeout=60s
./scripts/copy-data.sh $NAMESPACE
kubectl apply -f k8s/ -n $NAMESPACE

# 5. Monitor deployment
kubectl rollout status deployment/templ-pipeline -n $NAMESPACE
kubectl get pods,svc,ingress -l app=templ-pipeline -n $NAMESPACE
```

## 📋 Files You Must Edit

1. **k8s/deployment.yaml** - Line ~35:
   ```yaml
   image: hub.cerit.io/YOUR-USERNAME/templ-pipeline:latest
   ```

2. **k8s/ingress.yaml** - Lines ~14, ~18:
   ```yaml
   hosts:
     - "YOUR-DOMAIN.dyn.cloud.e-infra.cz"
   secretName: YOUR-DOMAIN-dyn-cloud-e-infra-cz-tls
   ```

## 🔍 Essential Monitoring Commands

```bash
# Check everything
kubectl get all -l app=templ-pipeline -n $NAMESPACE

# Check logs
kubectl logs -f deployment/templ-pipeline -n $NAMESPACE

# Debug pod
kubectl describe pod -l app=templ-pipeline -n $NAMESPACE

# Test locally
kubectl port-forward svc/templ-pipeline-svc 8080:80 -n $NAMESPACE
```

## 🌐 Your App URL
After deployment: `https://YOUR-DOMAIN.dyn.cloud.e-infra.cz`

## 📞 Quick Help
- CERIT Docs: https://docs.cerit.io/
- Harbor: https://hub.cerit.io/
- Rancher: https://rancher.cloud.e-infra.cz/
