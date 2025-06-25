# Optimized Dockerfile for minimal data deployment
FROM python:3.10-slim AS builder

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        build-essential \
        && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal requirements first for better caching
COPY requirements-server.txt ./

# Install Python dependencies with memory optimization
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --disable-pip-version-check \
        --no-warn-script-location \
        -r requirements-server.txt

# Copy project files (excluding large data via .dockerignore)
COPY . .

# Decompress SDF file for runtime use
RUN cd data-minimal/ligands && \
    gunzip processed_ligands_new.sdf.gz && \
    ls -lah

# Verify essential data files exist
RUN ls -lah data-minimal/embeddings/ && \
    ls -lah data-minimal/ligands/ && \
    echo "✅ Essential data files verified"

# Runtime stage - minimal
FROM python:3.10-slim

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git-lfs \
        curl \
        && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Create symlink for backward compatibility
RUN ln -sf /app/data-minimal /app/data

# Environment variables
ENV PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONPATH=/app

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/?healthz || exit 1

CMD ["streamlit", "run", "templ_pipeline/ui/app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--browser.gatherUsageStats", "false"]
