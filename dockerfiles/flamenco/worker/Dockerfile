# Stage 1: Downloader and extractor stage
FROM python:3.11-alpine AS downloader

# Install necessary dependencies for downloading Flamenco
RUN apk add --no-cache wget tar

# Download and extract the precompiled Flamenco package (manager and worker)
WORKDIR /tmp
RUN wget https://flamenco.blender.org/downloads/flamenco-3.5-linux-amd64.tar.gz -O flamenco.tar.gz && \
    tar -xzf flamenco.tar.gz && \
    ls -l /tmp/flamenco-3.5-linux-amd64

# Stage 2: Final image
FROM python:3.11-alpine

# Install necessary dependencies (including Blender headless)
RUN apk add --no-cache blender

# Copy Flamenco files from the downloader stage
COPY --from=downloader /tmp/flamenco-3.5-linux-amd64 /opt/flamenco

# Clean up unnecessary files from Flamenco installation
RUN rm -rf /opt/flamenco/{CHANGELOG.md,README.md,LICENSE} && \
    rm -rf /usr/share/blender/{tests,scripts,presets,docs}

# Set the working directory for Flamenco Worker
WORKDIR /opt/flamenco

# Expose the worker's port (if needed)
EXPOSE 8080

# Set the command to run the Flamenco Worker
CMD ["/opt/flamenco/flamenco-worker"]
