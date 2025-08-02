# TEMPL Pipeline Security Setup Guide

## 🔒 **CRITICAL SECURITY NOTICE**

This guide covers secure deployment of TEMPL Pipeline with proper credential management. **DO NOT** commit actual secrets to git repositories.

## 📋 **Prerequisites**

- Kubernetes cluster access
- `kubectl` configured
- `htpasswd` utility (install with `sudo apt-get install apache2-utils`)

## 🚀 **Quick Setup**

### 1. Generate Authentication Credentials

```bash
# Navigate to deployment scripts
cd deploy/scripts

# Generate authentication for both app and ingress
./generate-auth.sh \
  --app-password "your_secure_app_password" \
  --ingress-user "admin" \
  --ingress-password "your_secure_ingress_password" \
  --namespace "your-namespace"
```

### 2. Create Kubernetes Secrets

The script above will output the YAML needed. Alternatively, create manually:

```bash
# Create the main application secrets
kubectl create secret generic templ-pipeline-secrets \
  --from-literal=TEMPL_PASSWORD_HASH="your_generated_hash" \
  --from-literal=STREAMLIT_SERVER_COOKIE_SECRET="$(openssl rand -hex 32)" \
  -n your-namespace

# Create ingress basic auth (optional)
kubectl create secret generic templ-basic-auth \
  --from-literal=auth="$(htpasswd -nb admin your_password)" \
  -n your-namespace
```

### 3. Deploy Application

```bash
# Apply the deployment with secrets configured
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl apply -f deploy/kubernetes/service.yaml
kubectl apply -f deploy/kubernetes/ingress.yaml
```

## 🔐 **Security Architecture**

### Authentication Layers

1. **Application Level**: Password-based authentication in Streamlit
2. **Ingress Level**: HTTP Basic Authentication (optional)
3. **TLS**: HTTPS encryption for all communication

### Secret Management

- **Application secrets**: Stored in `templ-pipeline-secrets`
- **Ingress auth**: Stored in `templ-basic-auth`
- **Environment variables**: Non-sensitive config in ConfigMap

## 📁 **File Structure**

```
deploy/kubernetes/
├── secrets.template.yaml    # Template showing required secrets
├── secrets.yaml            # Actual secrets (git-ignored)
├── configmap.yaml          # Non-sensitive configuration
├── deployment.yaml         # References secrets properly
├── service.yaml            # Service definition
└── ingress.yaml            # Ingress with optional basic auth
```

## 🛡️ **Security Best Practices**

### Password Requirements
- Minimum 12 characters
- Include uppercase, lowercase, numbers, symbols
- Avoid dictionary words
- Unique passwords for each environment

### Secret Rotation
```bash
# Rotate application password
./generate-auth.sh --app-password "new_password"
kubectl patch secret templ-pipeline-secrets \
  -p='{"data":{"TEMPL_PASSWORD_HASH":"new_base64_hash"}}'

# Restart deployment to pick up new secrets
kubectl rollout restart deployment/templ-pipeline
```

### Monitoring
- Enable audit logging for secret access
- Monitor authentication failures
- Set up alerts for repeated failed login attempts

## 🔧 **Environment-Specific Setup**

### Development
```bash
export TEMPL_PASSWORD_HASH="your_dev_hash"
export STREAMLIT_SERVER_COOKIE_SECRET="dev_cookie_secret"
python scripts/run_streamlit_app.py
```

### Production
- Use Kubernetes secrets exclusively
- Enable all security features
- Regular security audits

### Staging
- Separate secrets from production
- Test secret rotation procedures
- Validate security configurations

## 🚨 **Troubleshooting**

### Common Issues

**Authentication fails after deployment:**
```bash
# Check if secrets exist
kubectl get secrets -l app=templ-pipeline

# Verify secret content
kubectl get secret templ-pipeline-secrets -o yaml

# Check pod logs
kubectl logs -l app=templ-pipeline
```

**Missing environment variables:**
```bash
# Verify ConfigMap
kubectl describe configmap templ-pipeline-config

# Check deployment environment
kubectl describe deployment templ-pipeline
```

### Validation Commands

```bash
# Test authentication module
cd templ_pipeline/ui/core
TEMPL_PASSWORD_HASH="your_hash" python auth.py

# Verify secret format
echo "your_password" | sha256sum
```

## 📚 **Additional Resources**

- [Kubernetes Secrets Documentation](https://kubernetes.io/docs/concepts/configuration/secret/)
- [Streamlit Security Best Practices](https://docs.streamlit.io/deploy/deploy-security)
- [CERIT-SC Security Guidelines](https://docs.cerit.io/docs/general/security-guide/)

## ⚠️ **IMPORTANT REMINDERS**

1. **NEVER** commit `secrets.yaml` to git
2. Use different passwords for each environment
3. Rotate secrets regularly (quarterly minimum)
4. Monitor authentication logs
5. Keep this documentation updated

---

*For additional security questions, consult your security team or CERIT-SC support.*