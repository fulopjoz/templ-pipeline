# Lightweight init container for downloading TEMPL Pipeline datasets
# Used to prepare data in persistent volumes before main application starts

FROM python:3.11-alpine

# Install required packages for data download and extraction
RUN apk add --no-cache \
    curl \
    wget \
    tar \
    gzip \
    && pip install --no-cache-dir requests zenodo-get

# Create working directory
WORKDIR /data-setup

# Copy the data setup scripts
COPY scripts/setup_pdbind_data.py /data-setup/
COPY deploy/scripts/init-data-setup.sh /data-setup/

# Make script executable
RUN chmod +x /data-setup/init-data-setup.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TEMPL_DATA_DIR=/app/data

# Entry point for data initialization
ENTRYPOINT ["/data-setup/init-data-setup.sh"]