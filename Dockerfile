# Optimized Dockerfile with lightweight embedding support
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
# Install PyTorch CPU first (largest dependency) for better caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --disable-pip-version-check \
        --no-warn-script-location \
        "torch>=1.10.0,<2.0.0" --index-url https://download.pytorch.org/whl/cpu && \
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

# Pre-download ESM2 model to reduce startup time (optional - adds ~2.6GB)
# Uncomment the next lines if you want to pre-cache the model
# RUN python -c "from transformers import EsmModel, EsmTokenizer; \
#     model_id='facebook/esm2_t33_650M_UR50D'; \
#     print('Downloading ESM2 model...'); \
#     EsmTokenizer.from_pretrained(model_id); \
#     EsmModel.from_pretrained(model_id); \
#     print('ESM2 model cached successfully')"

# Runtime stage - minimal
FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git-lfs \
        curl \
        libxrender1 \
        libxext6 \
        libsm6 \
        libxft2 \
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

# Copy startup script with explicit permissions
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh && \
    echo "✅ Startup script permissions:" && \
    ls -la /app/start.sh

# Environment variables - minimal set, let startup script handle Streamlit config
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface

# Create cache directories
RUN mkdir -p /app/.cache/transformers /app/.cache/huggingface

# Force rebuild by adding build timestamp
RUN echo "Build timestamp: $(date)" > /app/build_info.txt

# Expose port (DigitalOcean will set PORT dynamically)
EXPOSE 8080

# Health check using Streamlit's built-in endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Use the startup script as entrypoint
ENTRYPOINT ["/app/start.sh"]
