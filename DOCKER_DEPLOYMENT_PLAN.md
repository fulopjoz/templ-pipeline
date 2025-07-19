# TEMPL Pipeline Docker Deployment Plan

## Project Overview

The TEMPL (Template-based Protein-Ligand) Pipeline is a comprehensive web application for pose prediction using template-based approaches. This document provides a complete Docker deployment strategy for the Streamlit-based web interface.

### Current Architecture
- **Main Application**: Streamlit web app at `templ_pipeline/ui/app.py`
- **Launcher**: `run_streamlit_app.py` with port management and health checks
- **Core Pipeline**: Template-based pose prediction with MCS alignment
- **Data Requirements**: ~3GB of protein embeddings and ligand structures
- **Dependencies**: RDKit, NumPy, Pandas, BioPython, Biotite, SpyRMSD, Streamlit

## Docker Implementation Strategy

### 1. Multi-Stage Dockerfile Architecture

#### Base Image Selection
```dockerfile
FROM continuumio/miniconda3:latest as base
```
**Rationale**: Conda provides better dependency management for scientific packages like RDKit, avoiding compilation issues.

#### Stage 1: System Dependencies
```dockerfile
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*
```

#### Stage 2: Python Environment
```dockerfile
# Create conda environment with scientific packages
COPY requirements.txt pyproject.toml ./
RUN conda create -n templ python=3.11 -y && \
    conda activate templ && \
    conda install -c conda-forge rdkit pandas numpy scipy scikit-learn biopython -y && \
    pip install --no-cache-dir -r requirements.txt
```

#### Stage 3: Application Setup
```dockerfile
# Copy application code
COPY templ_pipeline/ ./templ_pipeline/
COPY run_streamlit_app.py ./
RUN conda activate templ && \
    pip install -e . --no-deps
```

#### Stage 4: Data Management
```dockerfile
# Download and cache large data files
RUN conda activate templ && \
    python -c "from templ_pipeline.core.data import download_required_data; download_required_data()"
```

### 2. Data Management Strategy

#### Large File Handling
The pipeline requires ~3GB of data files:
- `data/embeddings/templ_protein_embeddings_v1.0.0.npz` (~2GB)
- `data/ligands/templ_processed_ligands_v1.0.0.sdf.gz` (~800MB)

**Options**:
1. **Build-time Download**: Include in Docker image (slower build, larger image)
2. **Runtime Download**: Download on first container start (faster build, network dependency)
3. **Volume Mount**: External data volume (recommended for production)

#### Recommended Approach
```dockerfile
# Create data volume mount point
VOLUME ["/app/data"]

# Download script for missing data
COPY scripts/download-data.sh /app/scripts/
RUN chmod +x /app/scripts/download-data.sh
```

### 3. Docker Compose Configuration

#### Development Setup
```yaml
version: '3.8'
services:
  templ-pipeline:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./temp:/app/temp
    environment:
      - TEMPL_ENV=development
      - STREAMLIT_SERVER_HEADLESS=true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

#### Production Setup
```yaml
version: '3.8'
services:
  templ-pipeline:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "80:8501"
    volumes:
      - templ_data:/app/data
      - templ_temp:/app/temp
    environment:
      - TEMPL_ENV=production
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s

volumes:
  templ_data:
  templ_temp:
```

### 4. Configuration Files

#### .dockerignore
```dockerignore
# Build artifacts
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Development files
.git/
.gitignore
.pytest_cache/
.coverage
htmlcov/
.tox/
.venv/
venv/
ENV/
env/

# Documentation
docs/
*.md
!README.md

# Test files
tests/
test_*/
*_test.py
test_*.py

# Temporary files
temp/
tmp/
*.tmp
*.log

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Large data files (handle separately)
data/embeddings/*.npz
data/ligands/*.sdf.gz
benchmark_workspace_*/
output/
memory-bank/

# Jupyter notebooks
*.ipynb
.ipynb_checkpoints/
```

#### Environment Configuration
```dockerfile
# Environment variables for different deployment modes
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
ENV STREAMLIT_GLOBAL_LOG_LEVEL=info

# TEMPL Pipeline specific
ENV TEMPL_DATA_DIR=/app/data
ENV TEMPL_TEMP_DIR=/app/temp
ENV TEMPL_WORKERS=auto
ENV PYTHONPATH=/app
```

### 5. Build Variants

#### Development Dockerfile
```dockerfile
FROM continuumio/miniconda3:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create conda environment
COPY requirements.txt pyproject.toml ./
RUN conda create -n templ python=3.11 -y && \
    echo "conda activate templ" >> ~/.bashrc && \
    /opt/conda/envs/templ/bin/pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install in development mode
RUN /opt/conda/envs/templ/bin/pip install -e .

# Create data and temp directories
RUN mkdir -p /app/data /app/temp

# Expose port
EXPOSE 8501

# Entry point
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["development"]
```

#### Production Dockerfile
```dockerfile
FROM continuumio/miniconda3:latest as builder

# Build stage - install dependencies
RUN apt-get update && apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./

# Create optimized conda environment
RUN conda create -n templ python=3.11 -y && \
    conda install -n templ -c conda-forge \
    rdkit pandas numpy scipy scikit-learn biopython streamlit -y && \
    /opt/conda/envs/templ/bin/pip install --no-cache-dir \
    biotite spyrmsd tabulate zenodo-get && \
    conda clean -afy

# Production stage
FROM continuumio/miniconda3:latest

# Copy conda environment
COPY --from=builder /opt/conda/envs/templ /opt/conda/envs/templ

# Install minimal system dependencies
RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application
COPY templ_pipeline/ ./templ_pipeline/
COPY run_streamlit_app.py ./
COPY pyproject.toml ./

# Install application
RUN /opt/conda/envs/templ/bin/pip install --no-deps -e .

# Create runtime user
RUN useradd -m -u 1000 templ && \
    chown -R templ:templ /app
USER templ

# Create directories
RUN mkdir -p /app/data /app/temp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Entry point
COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["production"]
```

### 6. Entry Point Script

#### docker-entrypoint.sh
```bash
#!/bin/bash
set -e

# Activate conda environment
source /opt/conda/bin/activate templ

# Set deployment mode
MODE=${1:-production}

echo "Starting TEMPL Pipeline in $MODE mode..."

# Data directory setup
if [ ! -d "/app/data/embeddings" ] || [ ! -d "/app/data/ligands" ]; then
    echo "Required data files not found. Downloading..."
    python -c "
from templ_pipeline.core.data import download_required_data
download_required_data('/app/data')
    " || {
        echo "Warning: Data download failed. Some features may not work."
    }
fi

# Environment-specific configuration
case "$MODE" in
    "development")
        export STREAMLIT_GLOBAL_LOG_LEVEL=debug
        export TEMPL_WORKERS=1
        ;;
    "production")
        export STREAMLIT_GLOBAL_LOG_LEVEL=info
        export TEMPL_WORKERS=auto
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

# Start application
echo "Starting Streamlit application..."
exec python run_streamlit_app.py
```

### 7. Build and Deployment Scripts

#### build.sh
```bash
#!/bin/bash
set -e

VERSION=${1:-latest}
MODE=${2:-production}

echo "Building TEMPL Pipeline Docker image..."
echo "Version: $VERSION"
echo "Mode: $MODE"

# Build image
docker build \
    -f Dockerfile.${MODE} \
    -t templ-pipeline:${VERSION} \
    -t templ-pipeline:latest \
    .

echo "Build completed successfully!"
echo "Image size:"
docker images templ-pipeline:${VERSION}
```

#### deploy.sh
```bash
#!/bin/bash
set -e

MODE=${1:-production}
PORT=${2:-8501}

echo "Deploying TEMPL Pipeline..."
echo "Mode: $MODE"
echo "Port: $PORT"

# Stop existing container
docker-compose -f docker-compose.${MODE}.yml down 2>/dev/null || true

# Start new deployment
docker-compose -f docker-compose.${MODE}.yml up -d

echo "Deployment completed!"
echo "Application available at: http://localhost:${PORT}"

# Show logs
docker-compose -f docker-compose.${MODE}.yml logs -f
```

### 8. Optimization Features

#### Build Optimization
- Multi-stage builds to reduce final image size
- Layer caching for faster rebuilds
- Conda environment optimization
- Selective file copying with .dockerignore

#### Runtime Optimization
- Memory limits and CPU constraints
- Health checks for container orchestration
- Non-root user for security
- Volume mounts for data persistence

#### Performance Tuning
```dockerfile
# Optimize Python and Streamlit
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_SERVER_RUN_ON_SAVE=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### 9. Data Volume Management

#### External Data Volume
```bash
# Create data volume
docker volume create templ_data

# Download data to volume
docker run --rm -v templ_data:/data \
    templ-pipeline:latest \
    python -c "from templ_pipeline.core.data import download_required_data; download_required_data('/data')"
```

#### Backup Strategy
```bash
# Backup data volume
docker run --rm -v templ_data:/data -v $(pwd):/backup \
    ubuntu tar czf /backup/templ_data_backup.tar.gz -C /data .

# Restore data volume
docker run --rm -v templ_data:/data -v $(pwd):/backup \
    ubuntu tar xzf /backup/templ_data_backup.tar.gz -C /data
```

### 10. Monitoring and Health Checks

#### Built-in Health Endpoint
The Streamlit app provides a built-in health check endpoint at `/_stcore/health`.

#### Custom Health Check
```python
# Add to app.py for enhanced health monitoring
@st.cache_data
def health_check():
    """Enhanced health check for Docker"""
    try:
        # Check core components
        from templ_pipeline.core import pipeline
        
        # Check data availability
        data_status = check_data_files()
        
        # Check memory usage
        import psutil
        memory_percent = psutil.virtual_memory().percent
        
        return {
            "status": "healthy",
            "data_files": data_status,
            "memory_usage": f"{memory_percent:.1f}%",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
```

### 11. Security Considerations

#### Image Security
```dockerfile
# Use non-root user
RUN useradd -m -u 1000 templ
USER templ

# Limit capabilities
RUN chmod -R 755 /app
```

#### Network Security
```yaml
# docker-compose.yml
networks:
  templ-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 12. CI/CD Integration

#### GitHub Actions Workflow
```yaml
name: Docker Build and Deploy
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: |
          docker build -t templ-pipeline:${{ github.sha }} .
          
      - name: Test Docker image
        run: |
          docker run --rm -d -p 8501:8501 --name test-container templ-pipeline:${{ github.sha }}
          sleep 30
          curl -f http://localhost:8501/_stcore/health
          docker stop test-container
          
      - name: Push to registry
        if: github.ref == 'refs/heads/main'
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push templ-pipeline:${{ github.sha }}
```

## Implementation Timeline

### Phase 1: Core Docker Setup (Week 1)
1. Create basic Dockerfile for development
2. Set up docker-compose for local development
3. Implement data download strategy
4. Test basic functionality

### Phase 2: Production Optimization (Week 2)
1. Create multi-stage production Dockerfile
2. Optimize image size and build time
3. Implement health checks and monitoring
4. Add security configurations

### Phase 3: Deployment Infrastructure (Week 3)
1. Create deployment scripts
2. Set up CI/CD pipeline
3. Implement backup and recovery procedures
4. Performance testing and optimization

### Phase 4: Documentation and Testing (Week 4)
1. Complete deployment documentation
2. Create user guides and troubleshooting
3. Comprehensive testing across environments
4. Final optimization and refinement

## Estimated Resource Requirements

### Development Environment
- **CPU**: 2 cores minimum
- **Memory**: 4GB minimum (8GB recommended)
- **Storage**: 10GB (includes data files)
- **Network**: Stable internet for data download

### Production Environment
- **CPU**: 4 cores recommended
- **Memory**: 8GB minimum (16GB for heavy usage)
- **Storage**: 20GB (includes data, logs, temp files)
- **Network**: High bandwidth for file uploads/downloads

## Expected Benefits

1. **Consistency**: Identical environment across development and production
2. **Scalability**: Easy horizontal scaling with container orchestration
3. **Portability**: Deploy anywhere Docker is supported
4. **Isolation**: Clean separation from host system
5. **Reproducibility**: Version-controlled deployment configuration
6. **Maintenance**: Simplified updates and rollbacks

## Risk Mitigation

1. **Large Data Files**: Volume mounting and download strategies
2. **Memory Usage**: Container limits and monitoring
3. **Build Time**: Multi-stage builds and layer caching
4. **Networking**: Health checks and proper port management
5. **Data Persistence**: Backup and recovery procedures

This comprehensive Docker deployment plan provides a production-ready containerization strategy for the TEMPL Pipeline, ensuring reliable, scalable, and maintainable deployment across different environments.


