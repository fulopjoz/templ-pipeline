# TEMPL Pipeline - DigitalOcean Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the TEMPL Pipeline to DigitalOcean App Platform with optimized production configuration.

## Prerequisites

- DigitalOcean account with App Platform access
- GitHub repository with the TEMPL Pipeline code
- Git LFS configured for large molecular data files (225MB)

## Deployment Architecture

### Resource Configuration
- **Instance Size**: `professional-xs` (2 vCPU, 4GB RAM)
- **Instance Count**: 1 (with auto-scaling capability)
- **Storage**: 20GB for molecular data and processing
- **Port**: 8080 (DigitalOcean standard)

### Health Monitoring
- **Health Check Endpoint**: `/?healthz`
- **Initial Delay**: 60 seconds (for molecular data loading)
- **Check Interval**: 10 seconds
- **Timeout**: 5 seconds

## Step-by-Step Deployment

### Step 1: Prepare Repository

1. **Ensure Git LFS is configured**:
   ```bash
   git lfs track "*.npz"
   git lfs track "*.sdf"
   git add .gitattributes
   git commit -m "Configure Git LFS for molecular data"
   ```

2. **Verify large files are tracked**:
   ```bash
   git lfs ls-files
   # Should show molecular data files like protein_embeddings_base.npz
   ```

3. **Push to GitHub**:
   ```bash
   git push origin speedrun
   ```

### Step 2: Create DigitalOcean App

1. **Login to DigitalOcean Console**
2. **Navigate to App Platform**
3. **Click "Create App"**
4. **Connect GitHub Repository**:
   - Repository: `fulopjoz/templ-pipeline`
   - Branch: `speedrun`
   - Auto-deploy: Enabled

### Step 3: Configure App Settings

#### Basic Configuration
- **App Name**: `templ-pipeline`
- **Environment**: Python
- **Build Command**: Automatic (uses Dockerfile)
- **Run Command**: 
  ```bash
  streamlit run templ_pipeline/ui/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false
  ```

#### Resource Settings
- **Instance Size**: Professional XS ($12/month)
- **Instance Count**: 1
- **Auto-scaling**: Disabled initially

#### Environment Variables
```yaml
STREAMLIT_SERVER_HEADLESS: "true"
STREAMLIT_SERVER_ENABLE_CORS: "false"
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION: "false"
STREAMLIT_BROWSER_GATHER_USAGE_STATS: "false"
PYTHONPATH: "/app"
STREAMLIT_THEME_BASE: "light"
```

#### Health Check Configuration
- **HTTP Path**: `/?healthz`
- **Initial Delay**: 60 seconds
- **Period**: 10 seconds
- **Timeout**: 5 seconds
- **Success Threshold**: 1
- **Failure Threshold**: 3

### Step 4: Deploy Application

1. **Review Configuration**
2. **Click "Create Resources"**
3. **Monitor Build Process**:
   - Build typically takes 5-10 minutes
   - Watch for Git LFS file downloads
   - Monitor dependency installation

### Step 5: Verify Deployment

#### Check Application Status
1. **Wait for "Running" status**
2. **Access public URL** (provided by DigitalOcean)
3. **Verify health endpoint**: `https://your-app.ondigitalocean.app/?healthz`

#### Test Core Functionality
1. **Load main interface**
2. **Test molecule input** (SMILES: `CCO`)
3. **Test protein input** (PDB ID: `1iky`)
4. **Verify pose prediction works**

## Troubleshooting

### Common Issues

#### Build Failures
- **Git LFS timeout**: Increase build timeout in app settings
- **Memory issues**: Upgrade to larger instance size temporarily
- **Dependency conflicts**: Check requirements.txt compatibility

#### Runtime Issues
- **Health check failures**: Verify endpoint responds with "OK"
- **Memory errors**: Monitor resource usage and upgrade if needed
- **Slow response**: Check molecular data loading times

#### Application Issues
- **Black page**: Verify JavaScript is enabled in browser
- **Import errors**: Check PYTHONPATH environment variable
- **Processing failures**: Monitor logs for dependency issues

### Log Analysis

#### Access Application Logs
1. **DigitalOcean Console** → **App Platform** → **Your App**
2. **Runtime Logs** tab
3. **Filter by severity** (Error, Warning, Info)

#### Key Log Patterns
- **Successful startup**: `"Hardware detected: gpu-large"`
- **Health check**: `"OK"` responses to health endpoint
- **Processing errors**: Look for RDKit or pipeline errors

### Performance Optimization

#### Resource Monitoring
- **CPU Usage**: Should be <50% during normal operation
- **Memory Usage**: Should be <3GB for typical workloads
- **Response Times**: <30 seconds for simple predictions

#### Scaling Recommendations
- **Light usage**: Professional XS (current)
- **Medium usage**: Professional S (4 vCPU, 8GB RAM)
- **Heavy usage**: Professional M (8 vCPU, 16GB RAM)

## Maintenance

### Regular Tasks
- **Monitor resource usage** weekly
- **Check application logs** for errors
- **Update dependencies** monthly
- **Test core functionality** after updates

### Update Deployment
1. **Push changes to speedrun branch**
2. **Auto-deployment triggers build**
3. **Monitor deployment in console**
4. **Verify functionality after update**

### Backup Strategy
- **Code**: Stored in GitHub repository
- **Configuration**: Documented in app.yaml
- **Data**: Molecular databases in Git LFS

## Cost Optimization

### Current Configuration Cost
- **Professional XS**: $12/month
- **Bandwidth**: ~$0.01/GB
- **Build minutes**: Included in plan

### Cost Reduction Options
- **Basic tier**: $5/month (1 vCPU, 512MB RAM) - may not support molecular processing
- **Optimize build**: Reduce dependencies to speed builds
- **Scale down**: Use smaller instance during low usage

## Security Considerations

### Automatic Security Features
- **HTTPS/SSL**: Automatically configured
- **DDoS protection**: Included with App Platform
- **Network isolation**: Container-level isolation

### Additional Security
- **Environment variables**: Never commit secrets to repository
- **Access control**: Use DigitalOcean teams for multi-user access
- **Monitoring**: Enable alerts for unusual activity

## Support Resources

### DigitalOcean Documentation
- [App Platform Documentation](https://docs.digitalocean.com/products/app-platform/)
- [Python App Deployment](https://docs.digitalocean.com/tutorials/app-deploy-python-django/)
- [Troubleshooting Guide](https://docs.digitalocean.com/products/app-platform/troubleshooting/)

### TEMPL Pipeline Support
- **Repository Issues**: [GitHub Issues](https://github.com/fulopjoz/templ-pipeline/issues)
- **Documentation**: README.md in repository
- **Scientific Support**: Contact repository maintainers

---

## Quick Reference

### Deployment Commands
```bash
# Deploy from local repository
git push origin speedrun

# Check deployment status
doctl apps list
doctl apps get <app-id>

# View logs
doctl apps logs <app-id> --type=deploy
doctl apps logs <app-id> --type=run
```

### Health Check URLs
- **Production**: `https://your-app.ondigitalocean.app/?healthz`
- **Local testing**: `http://localhost:8501/?healthz`

### Key Configuration Files
- `app.yaml` - DigitalOcean App Platform configuration
- `Dockerfile` - Container build instructions
- `requirements.txt` - Python dependencies
- `run_streamlit_app.py` - Application launcher 