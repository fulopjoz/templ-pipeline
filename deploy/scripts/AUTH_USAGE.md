# Authentication Script Usage Guide

## ✅ Fixed Authentication Script

The `generate-auth.sh` script has been fixed and is now working correctly!

## 🚀 Quick Usage

### Generate App Password Only
```bash
# Just press Enter when prompted for ingress password
./deploy/scripts/generate-auth.sh --app-password "YourSecurePassword2025"
```

### Generate Both App and Ingress Authentication
```bash
./deploy/scripts/generate-auth.sh \
  --app-password "YourSecurePassword2025" \
  --ingress-user "admin" \
  --ingress-password "IngressPassword2025"
```

### With Custom Namespace
```bash
./deploy/scripts/generate-auth.sh \
  --app-password "YourSecurePassword2025" \
  --namespace "your-namespace"
```

## 📋 What the Script Generates

### App Authentication
The script outputs:
```
---YAML START---
env:
- name: TEMPL_PASSWORD_HASH
  value: "generated_sha256_hash"
---YAML END---
```

**Copy this to your deployment.yaml:**
```yaml
spec:
  containers:
  - name: templ-pipeline
    image: cerit.io/xfulop/templ-pipeline:latest
    env:
    - name: TEMPL_PASSWORD_HASH
      value: "your_generated_hash"
```

### Ingress Authentication (Optional)
If you provide ingress password, you get:
```
---COMMAND START---
kubectl create secret generic templ-basic-auth --from-literal=auth='user:hashedpassword'
---COMMAND END---
```

## 🔧 Complete Deployment Example

1. **Generate credentials:**
```bash
./deploy/scripts/generate-auth.sh --app-password "MySecurePassword2025"
```

2. **Copy the YAML output and add to your deployment.yaml**

3. **Build and push secure image:**
```bash
./deploy/scripts/build.sh latest xfulop true
```

4. **Deploy with authentication:**
```bash
kubectl apply -f deploy/kubernetes/deployment.yaml -n your-namespace
kubectl apply -f deploy/kubernetes/service.yaml -n your-namespace
kubectl apply -f deploy/kubernetes/ingress-secure.yaml -n your-namespace
```

## ✅ Verification

Your deployed app should now:
- Show a password login screen
- Accept your chosen password
- Provide access to the TEMPL Pipeline interface after authentication

## 🔒 Security Notes

- **Change Default Password**: Never use `templ2025` in production
- **Strong Passwords**: Use 12+ characters with mixed case, numbers, symbols
- **Store Securely**: Use Kubernetes secrets for sensitive data
- **Regular Rotation**: Change passwords quarterly

The authentication script is now ready for production use! 🚀