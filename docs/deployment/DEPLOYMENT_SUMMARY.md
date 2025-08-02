# TEMPL Pipeline CERIT Deployment - Complete Guide Summary

This document provides an overview of all deployment options and guides for deploying the TEMPL Pipeline to CERIT infrastructure.

## 📚 Available Deployment Guides

### 1. **Complete Step-by-Step Solution** (`STEP_BY_STEP_SOLUTION.md`)
- **Best for**: First-time deployment or detailed understanding
- **Content**: Comprehensive guide with all phases, troubleshooting, and verification
- **Use when**: You want to understand every step and have time for detailed deployment

### 2. **Updated CERIT Deployment Guide** (`CERIT_DEPLOYMENT_UPDATED.md`)
- **Best for**: Reference guide with all options
- **Content**: Complete reference with prerequisites, troubleshooting, and maintenance
- **Use when**: You need a comprehensive reference for ongoing deployment management

### 3. **Quick Reference Guide** (`QUICK_DEPLOY_REFERENCE.md`)
- **Best for**: Experienced users or quick deployment
- **Content**: Command reference and common scenarios
- **Use when**: You know the process and need quick command reference

## 🚀 Deployment Options

### Option 1: One-Command Deployment (Recommended)
```bash
# Complete automated deployment
./deploy/scripts/deploy-complete.sh -u your-harbor-username -n your-namespace -d your-domain
```

**Example:**
```bash
./deploy/scripts/deploy-complete.sh -u johndoe -n johndoe-ns -d templ-johndoe
```

### Option 2: Manual Step-by-Step Deployment
Follow the complete guide in `STEP_BY_STEP_SOLUTION.md` for detailed control over each step.

### Option 3: Quick Deployment
Use the quick reference in `QUICK_DEPLOY_REFERENCE.md` for experienced users.

## 📋 Prerequisites Summary

### Required Information
| Item | Source | Example |
|------|--------|---------|
| **Harbor Username** | [hub.cerit.io](https://hub.cerit.io/) | `johndoe@university.cz` |
| **CLI Secret** | Harbor Profile → CLI Secret | `abc123def456` |
| **Namespace** | [rancher.cloud.e-infra.cz](https://rancher.cloud.e-infra.cz/) | `johndoe-ns` |
| **Domain Name** | Choose unique name | `templ-johndoe` |

### Required Tools
- Docker (for building and pushing images)
- kubectl (configured for CERIT cluster)
- Git (for repository access)

## 🎯 What You'll Deploy

Your TEMPL Pipeline includes:

### Core Components
- **Streamlit Web Interface**: Interactive UI for protein-ligand pose prediction
- **CLI Tools**: Command-line interface for batch processing
- **Scientific Libraries**: RDKit, Biopython, scikit-learn, and more
- **Data Management**: Persistent storage for molecular data (~3GB)

### Infrastructure Features
- **Security**: HTTPS with automatic Let's Encrypt certificates
- **Monitoring**: Health checks and comprehensive logging
- **Scalability**: Kubernetes-based deployment with resource limits
- **Persistence**: NFS-backed persistent volumes for data storage

## 🔧 Repository Structure

```
templ_pipeline/
├── deploy/                          # Deployment files
│   ├── docker/                      # Docker configuration
│   │   ├── Dockerfile               # Multi-stage build
│   │   ├── docker-entrypoint.sh    # Container startup
│   │   └── docker-compose.yml      # Local development
│   ├── kubernetes/                  # Kubernetes manifests
│   │   ├── deployment.yaml         # Main application
│   │   ├── service.yaml            # Service definition
│   │   ├── ingress.yaml            # External access
│   │   └── pvc.yaml                # Persistent storage
│   └── scripts/                     # Deployment automation
│       ├── build.sh                # Docker build
│       ├── deploy.sh               # Kubernetes deployment
│       ├── copy-data.sh            # Data transfer
│       └── deploy-complete.sh      # One-command deployment
├── templ_pipeline/                  # Application source
│   ├── ui/                         # Streamlit interface
│   └── cli/                        # Command-line tools
└── scripts/                         # Utility scripts
    └── run_streamlit_app.py        # Streamlit launcher
```

## 🌐 Access and URLs

### Application Access
- **Main URL**: `https://your-domain.dyn.cloud.e-infra.cz`
- **Health Check**: `https://your-domain.dyn.cloud.e-infra.cz/_stcore/health`

### Management URLs
- **CERIT Documentation**: https://docs.cerit.io/
- **Harbor Registry**: https://hub.cerit.io/
- **Rancher Dashboard**: https://rancher.cloud.e-infra.cz/
- **e-INFRA Cloud**: https://docs.e-infra.cz/cloud/einfracz-cloud/

## 📊 Resource Requirements

### Minimum Configuration
- **CPU**: 500m (0.5 cores)
- **Memory**: 2Gi
- **Storage**: 5Gi for data + 2Gi for temp files

### Recommended Configuration
- **CPU**: 2 cores
- **Memory**: 4Gi
- **Storage**: 10Gi for data + 5Gi for temp files

## 🔒 Security Features

1. **Non-root User**: Container runs as user ID 1000
2. **Read-only Data**: Data volume mounted read-only
3. **Security Context**: Follows CERIT security policies
4. **TLS**: Automatic HTTPS via Let's Encrypt
5. **Resource Limits**: CPU and memory limits prevent resource exhaustion

## 📈 Monitoring and Maintenance

### Health Checks
```bash
# View logs
kubectl logs -f deployment/templ-pipeline -n your-namespace

# Check status
kubectl get all -l app=templ-pipeline -n your-namespace

# Test health
curl https://your-domain.dyn.cloud.e-infra.cz/_stcore/health
```

### Updates
```bash
# Update application
./deploy/scripts/build.sh v1.1.0 your-username
docker push hub.cerit.io/your-username/templ-pipeline:v1.1.0
kubectl set image deployment/templ-pipeline templ-pipeline=hub.cerit.io/your-username/templ-pipeline:v1.1.0 -n your-namespace

# Update data
./deploy/scripts/copy-data.sh your-namespace
kubectl rollout restart deployment/templ-pipeline -n your-namespace
```

## 🧹 Cleanup

```bash
# Remove deployment
kubectl delete -f deploy/kubernetes/ -n your-namespace

# Remove data (optional)
kubectl delete pvc templ-data-pvc -n your-namespace
```

## ✅ Success Checklist

After deployment, verify:

- [ ] Application accessible at `https://your-domain.dyn.cloud.e-infra.cz`
- [ ] SSL certificate issued (green lock in browser)
- [ ] Streamlit interface loads correctly
- [ ] Health endpoint responds
- [ ] Pod logs show no errors
- [ ] Data files accessible in `/app/data/`

## 🎉 Expected Results

Once deployed, your TEMPL Pipeline provides:

1. **Web Interface**: Interactive Streamlit UI for protein-ligand pose prediction
2. **CLI Tools**: Command-line interface for batch processing
3. **Data Management**: Persistent storage for molecular data
4. **Security**: HTTPS with automatic certificate renewal
5. **Monitoring**: Health checks and comprehensive logging
6. **Scalability**: Kubernetes-based deployment with resource limits
7. **Scientific Computing**: RDKit, Biopython, and other scientific libraries

## 📞 Support

For issues specific to:
- **CERIT Platform**: Contact CERIT support
- **TEMPL Pipeline**: Check project documentation or repository issues
- **Kubernetes**: Refer to Kubernetes documentation

## 🚀 Quick Start

For immediate deployment:

1. **Get your CERIT information** (Harbor username, namespace, domain)
2. **Run one-command deployment**:
   ```bash
   ./deploy/scripts/deploy-complete.sh -u your-username -n your-namespace -d your-domain
   ```
3. **Wait 5-10 minutes** for SSL certificate and DNS propagation
4. **Access your application** at `https://your-domain.dyn.cloud.e-infra.cz`

Your TEMPL Pipeline is now ready for scientific research and collaboration! 🎉 