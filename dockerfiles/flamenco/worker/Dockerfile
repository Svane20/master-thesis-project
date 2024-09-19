# Stage 1: Downloader and extractor stage
FROM ubuntu:22.04 AS downloader

# Install necessary dependencies for downloading Blender and Flamenco
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the Blender version to download
ENV BLENDER_VERSION=4.1.1
ENV BLENDER_VERSION_PREFIX=4.1

# Download and extract the specified Blender version using the version prefix
WORKDIR /tmp
RUN wget https://download.blender.org/release/Blender${BLENDER_VERSION_PREFIX}/blender-${BLENDER_VERSION}-linux-x64.tar.xz -O blender.tar.xz && \
    tar -xf blender.tar.xz && \
    mv blender-${BLENDER_VERSION}-linux-x64 blender

# Download and extract the precompiled Flamenco package (manager and worker)
RUN wget https://flamenco.blender.org/downloads/flamenco-3.5-linux-amd64.tar.gz -O flamenco.tar.gz && \
    tar -xzf flamenco.tar.gz

# Stage 2: Final image
FROM ubuntu:22.04

# Install necessary dependencies for Blender and Flamenco Worker
RUN apt-get update && apt-get install -y \
    unzip \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-utils \
    libxrender1 \
    libxext6 \
    libxi6 \
    libxkbcommon0 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libpulse0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Blender files from the downloader stage
COPY --from=downloader /tmp/blender /opt/blender

# Copy Flamenco files from the downloader stage
COPY --from=downloader /tmp/flamenco-3.5-linux-amd64 /opt/flamenco

# Copy Blender Flamenco add-on from the local context (make sure this file is in the same directory as your Dockerfile)
COPY flamenco-addon.zip /opt/flamenco-addon.zip

# Install Flamenco add-on into Blender
RUN mkdir -p /root/.config/blender/${BLENDER_VERSION}/scripts/addons

# Create a symbolic link to make Blender available globally
RUN ln -s /opt/blender/blender /usr/bin/blender

# Set default environment variables for the worker
ENV FLAMENCO_MANAGER_URL="http://flamenco-manager:8080"
ENV FLAMENCO_JOB_STORAGE="/var/flamenco/output"

# Enable and configure the Flamenco add-on
RUN blender --background --python-expr "import bpy; bpy.ops.preferences.addon_install(filepath='/opt/flamenco-addon.zip'); bpy.ops.preferences.addon_enable(module='flamenco'); is_installed = 'flamenco' in bpy.context.preferences.addons; prefs = bpy.context.preferences.addons['flamenco'].preferences if is_installed else None; print(f'Flamenco add-on is installed: {is_installed}'); prefs.manager_url, prefs.job_storage = ('${FLAMENCO_MANAGER_URL}', '${FLAMENCO_JOB_STORAGE}') if prefs else (None, None); bpy.ops.wm.save_userpref() if prefs else None; print(f'Flamenco Manager URL: {prefs.manager_url}' if prefs else ''); print(f'Flamenco Job Storage: {prefs.job_storage}' if prefs else '');"

# Check installed addons
RUN blender --background --python-expr "import bpy; print('Installed add-ons:', bpy.context.preferences.addons.keys())"

# Clean up unnecessary files from Flamenco installation
RUN rm -rf /opt/flamenco/{CHANGELOG.md,README.md,LICENSE}

# Set the working directory for Flamenco Worker
WORKDIR /opt/flamenco

# Expose the worker's port (if needed)
EXPOSE 8080

# Set the command to run the Flamenco Worker
CMD ["/opt/flamenco/flamenco-worker"]
