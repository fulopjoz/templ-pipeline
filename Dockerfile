# --- build stage -------------------------------------------------
FROM python:3.10-slim AS builder

# Install git and git-lfs so that large embeddings stored via LFS are fetched
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs build-essential && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Fetch real LFS objects (embeddings, large SDFs, etc.)
RUN git lfs pull

# --- runtime stage ----------------------------------------------
FROM python:3.10-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends git-lfs curl && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONPATH=/app

EXPOSE 8080

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/?healthz || exit 1

CMD ["streamlit", "run", "templ_pipeline/ui/app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--browser.gatherUsageStats", "false"] 