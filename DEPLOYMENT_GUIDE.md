# TEMPL Pipeline Deployment Guide

## ✅ Recent Fixes Applied

### Fixed Issues:
1. **Dockerfile CMD Path** - Fixed path from `ui/app.py` to `templ_pipeline/ui/app.py`
2. **App Platform Config** - Fixed dockerfile path and increased memory allocation
3. **Resource Allocation** - Upgraded from 1GB to 4GB RAM (2vCPU)

## 🚨 Critical Pre-Deployment Checklist

### 1. Git LFS Authentication Setup

The app requires large data files (protein embeddings ~90MB, ligands ~63MB) stored with Git LFS:

```bash
# Verify LFS files are available locally
git lfs ls-files
git lfs pull

# Check file sizes (should be > 5MB, not ~130 bytes pointers)
ls -lh data/embeddings/protein_embeddings_base.npz
ls -lh data/ligands/processed_ligands_new_unzipped.sdf
```

**For DigitalOcean App Platform:**
- Ensure your GitHub repository has LFS properly configured
- The build process runs `git lfs pull` automatically
- If issues persist, consider pre-extracting LFS files to regular Git

### 2. Memory Requirements Validation

**Current Allocation:** 4GB RAM (2vCPU) - `apps-s-2vcpu-4gb-fixed`

**Minimum Requirements:**
- **2GB RAM** - Basic functionality
- **4GB RAM** - Recommended for production
- **8GB RAM** - Optimal performance with multiple users

**Memory Usage Breakdown:**
- App Framework: ~500MB
- Protein embeddings: ~400MB  
- Ligand database: ~200MB
- Pytorch/Transformers: ~800MB
- Runtime overhead: ~1GB
- **Total: ~2.9GB**

### 3. CPU vs GPU Configuration

**Current Setup:** CPU-only deployment (recommended for App Platform)

**Dependencies Status:**
- ✅ CPU-optimized versions of PyTorch/Transformers
- ⚠️ Some GPU packages may be installed but unused
- ✅ ESM2 model works efficiently on CPU

### 4. Environment Variables (Optional)

Set these in DigitalOcean App Platform if needed:

```bash
# Override default paths if necessary
TEMPL_EMBEDDING_PATH=/app/data/embeddings/protein_embeddings_base.npz
TEMPL_LIGANDS_PATH=/app/data/ligands/processed_ligands_new_unzipped.sdf

# Performance tuning
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
```

## 🧪 Pre-Deployment Testing

### Test Build Locally

```bash
# Build the Docker image
docker build -t templ-pipeline .

# Test the app locally
docker run -p 8080:8080 templ-pipeline

# Visit http://localhost:8080 to verify
```

### Validate Critical Files

```bash
# Check that critical data files exist and have correct sizes
docker run --rm templ-pipeline ls -lh /app/data/embeddings/
docker run --rm templ-pipeline ls -lh /app/data/ligands/

# Verify app starts without errors
docker run --rm templ-pipeline python -c "import templ_pipeline.ui.app; print('App imports successfully')"
```

## 🚀 Deployment Steps

### Method 1: DigitalOcean App Platform (Recommended)

1. **Push fixes to your repository**
   ```bash
   git add Dockerfile templ-app.yaml
   git commit -m "Fix deployment configuration paths and resources"
   git push origin main
   ```

2. **Deploy via App Platform**
   ```bash
   # Using doctl CLI
   doctl apps create --spec templ-app.yaml

   # Or via web interface - upload templ-app.yaml
   ```

3. **Monitor deployment**
   - Watch build logs for LFS file downloading
   - Check memory usage after startup
   - Test a sample prediction

### Method 2: Docker Droplet (Alternative)

If App Platform has limitations, use a Docker-enabled droplet:

```bash
# Create droplet with Docker pre-installed
doctl compute droplet create templ-production \
  --region fra1 \
  --size s-2vcpu-4gb \
  --image docker-20-04

# Deploy via docker-compose or direct docker run
```

## 🔍 Post-Deployment Monitoring

### Health Checks

1. **App Startup** - Should be ready within 2-3 minutes
2. **Memory Usage** - Monitor RAM consumption
3. **Response Times** - Test with sample predictions
4. **Error Logs** - Check for LFS or dependency issues

### Performance Benchmarks

- **Simple prediction**: < 30 seconds
- **Complex prediction (100+ conformers)**: 2-5 minutes  
- **Memory baseline**: ~1.5GB after startup
- **Peak memory**: ~3GB during heavy processing

## ⚠️ Known Limitations

1. **Single-user optimization** - App Platform best for limited concurrent users
2. **Processing time** - Complex predictions take several minutes
3. **File upload limits** - Large protein files may need size limit adjustments
4. **Cold starts** - First request after idle period slower

## 🆘 Troubleshooting

### Common Issues:

**App won't start:**
- Check LFS files downloaded correctly (`git lfs pull`)
- Verify Dockerfile path is correct
- Ensure memory allocation is adequate

**Out of memory errors:**
- Upgrade to larger instance size
- Monitor memory usage patterns
- Consider optimizing concurrent processing

**Missing data files:**
- Verify Git LFS setup
- Check data directory structure
- Ensure embedding/ligand files present

**Slow performance:**
- Consider CPU-optimized instance types
- Reduce default conformer counts
- Monitor worker process allocation

## 📞 Support

For deployment issues:
1. Check DigitalOcean build logs
2. Verify local Docker build succeeds
3. Test critical imports and file access
4. Monitor memory and CPU usage

---

**Status: READY FOR DEPLOYMENT** ✅

With the applied fixes, the app should deploy successfully on DigitalOcean App Platform. 