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

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Environment variables (Streamlit will use STREAMLIT_* env vars set in start.sh)
ENV PYTHONPATH=/app

# Expose port (DigitalOcean will set PORT dynamically)
EXPOSE 8080

# Health check - use a simple approach that works with dynamic ports
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/_stcore/health || exit 1

# Use the startup script
CMD ["/app/start.sh"]
