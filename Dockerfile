# Multi-stage Dockerfile for TEMPL Pipeline
# Optimized for CERIT platform deployment with scientific computing dependencies

#==============================================================================
# Stage 1: Base Builder - Install system dependencies and conda environment
#==============================================================================
FROM continuumio/miniconda3:latest as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Create optimized conda environment for scientific computing
RUN conda create -n templ python=3.11 -y && \
    conda install -n templ -c conda-forge \
    rdkit=2024.09.6 pandas numpy scipy scikit-learn biopython streamlit -y && \
    /opt/conda/envs/templ/bin/pip install --no-cache-dir \
    biotite spyrmsd tabulate zenodo-get rich tqdm colorama pebble psutil && \
    conda clean -afy

#==============================================================================
# Stage 2: Production Image - Lightweight runtime environment
#==============================================================================
FROM continuumio/miniconda3:latest

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from builder
COPY --from=builder /opt/conda/envs/templ /opt/conda/envs/templ

# Set working directory
WORKDIR /app

# Copy application files
COPY templ_pipeline/ ./templ_pipeline/
COPY run_streamlit_app.py ./
COPY pyproject.toml ./
COPY setup_templ_env.sh ./

# Install application in development mode
RUN /opt/conda/envs/templ/bin/pip install --no-deps -e .

# Create non-root user for security
RUN useradd -m -u 1000 templ && \
    chown -R templ:templ /app

# Create directories for data and temporary files
RUN mkdir -p /app/data /app/temp && \
    chown -R templ:templ /app/data /app/temp

# Copy and make executable the entry point script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Switch to non-root user
USER templ

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
ENV STREAMLIT_GLOBAL_LOG_LEVEL=info
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_SERVER_RUN_ON_SAVE=false
ENV TEMPL_DATA_DIR=/app/data
ENV TEMPL_TEMP_DIR=/app/temp
ENV TEMPL_WORKERS=auto
ENV PYTHONPATH=/app

# Create volume mount points for data persistence
VOLUME ["/app/data", "/app/temp"]

# Health check using Streamlit's built-in endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose Streamlit port
EXPOSE 8501

# Set entry point
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["production"]