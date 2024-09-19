# Stage 1: Downloader and extractor stage
FROM python:3.11-alpine AS downloader

# Install necessary dependencies for downloading Flamenco
RUN apk add --no-cache wget tar

# Set work directory and download Flamenco
WORKDIR /tmp
RUN wget https://flamenco.blender.org/downloads/flamenco-3.5-linux-amd64.tar.gz -O flamenco.tar.gz && \
    tar -xzf flamenco.tar.gz && \
    rm flamenco.tar.gz

# Stage 2: Final image
FROM python:3.11-alpine

# Install only necessary runtime dependencies
RUN apk add --no-cache libstdc++ libc6-compat

# Copy Flamenco files from the downloader stage
COPY --from=downloader /tmp/flamenco-3.5-linux-amd64 /opt/flamenco

# Clean up unnecessary files from Flamenco installation
RUN rm -rf /opt/flamenco/{CHANGELOG.md,README.md,LICENSE}

# Set the working directory for Flamenco Manager
WORKDIR /opt/flamenco

# Expose the manager's port
EXPOSE 8080

# Set the command to run the Flamenco Manager
CMD ["/opt/flamenco/flamenco-manager"]