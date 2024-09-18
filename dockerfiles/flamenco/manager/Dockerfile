# Use a slim Python base image
FROM python:3.11-slim

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and extract the precompiled Flamenco package (manager and worker)
RUN wget https://flamenco.blender.org/downloads/flamenco-3.5-linux-amd64.tar.gz -O /tmp/flamenco.tar.gz && \
    tar -xzf /tmp/flamenco.tar.gz -C /opt/ && \
    rm /tmp/flamenco.tar.gz

# Set the working directory for Flamenco Manager
WORKDIR /opt/flamenco

# Expose the manager's port
EXPOSE 8080

# Set the command to run the Flamenco Manager
CMD ["./flamenco-manager", "--config", "/opt/flamenco/config.yaml"]